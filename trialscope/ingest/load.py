"""Load landing files into the DuckDB `raw` schema (idempotent upsert by nct_id).

raw.studies always holds the latest known version of each trial; history is captured
afterwards by the dbt snapshot (SCD2). raw._loaded_files makes re-runs no-ops.

    python -m trialscope.ingest.load            # load every landing file not loaded yet
"""
from __future__ import annotations

import gzip
import json
import time

import duckdb
import pandas as pd

from trialscope import config
from trialscope.ingest.parse import parse_study

CHILD_TABLES = ["conditions", "condition_mesh", "browse_branches", "interventions", "intervention_mesh", "sites", "references"]
BATCH = 20000


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    from trialscope.db import connect as _connect

    return _connect(read_only)


def _upsert(con, table: str, df: pd.DataFrame, ncts: pd.DataFrame) -> None:
    exists = con.execute("select count(*) from information_schema.tables where table_schema='raw' and table_name=?",
                         [table]).fetchone()[0]
    con.register("batch_df", df)
    if not exists:
        con.execute(f"create table raw.{table} as select * from batch_df")
    else:
        con.execute(f"delete from raw.{table} where nct_id in (select nct_id from batch_ncts)")
        con.execute(f"insert into raw.{table} by name select * from batch_df")
    con.unregister("batch_df")


def load_batch(con, studies: list[dict], source: str) -> int:
    parsed = [parse_study(s) for s in studies]
    frames = {t: pd.DataFrame([r for p in parsed for r in p[t]]) for t in ["studies"] + CHILD_TABLES}
    st = frames["studies"]
    if st.empty:
        return 0
    # several versions of one trial in a batch: keep the most recently updated
    st = st.sort_values("last_update_date").drop_duplicates("nct_id", keep="last")
    st["last_update_ts"] = pd.to_datetime(st.last_update_date).astype("datetime64[us]")   # SCD2 valid_from = registry update date
    st["source_file"], st["loaded_at"] = source, pd.Timestamp.now().floor("us")
    ncts = st[["nct_id"]]
    con.register("batch_ncts", ncts)
    _upsert(con, "studies", st, ncts)
    for t in CHILD_TABLES:
        df = frames[t]
        if df.empty:
            con.execute(f"delete from raw.{t} where nct_id in (select nct_id from batch_ncts)") if _table_exists(con, t) else None
            continue
        df = df[df.nct_id.isin(set(ncts.nct_id))].drop_duplicates()
        _upsert(con, t, df, ncts)
    con.unregister("batch_ncts")
    return len(st)


def _table_exists(con, t: str) -> bool:
    return con.execute("select count(*) from information_schema.tables where table_schema='raw' and table_name=?", [t]).fetchone()[0] > 0


def load_all() -> int:
    con = connect()
    con.execute("create schema if not exists raw")
    con.execute("create table if not exists raw._loaded_files(file varchar primary key, n_studies integer, loaded_at timestamp)")
    done = {r[0] for r in con.execute("select file from raw._loaded_files").fetchall()}
    files = sorted(p for p in config.LANDING_DIR.glob("*.jsonl.gz") if p.name not in done)
    total = 0
    for path in files:
        state = config.LANDING_DIR / "_state.json"
        if state.exists() and json.loads(state.read_text()).get("in_progress") and json.loads(state.read_text())["file"] == path.name:
            print(f"skip {path.name} (pull still in progress — re-run the pull to finish it)")
            continue
        t0, n, buf = time.time(), 0, []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    buf.append(json.loads(line))
                if len(buf) >= BATCH:
                    n += load_batch(con, buf, path.name)
                    buf = []
        n += load_batch(con, buf, path.name)
        con.execute("insert into raw._loaded_files values (?, ?, now())", [path.name, n])
        print(f"loaded {path.name}: {n:,} studies in {time.time() - t0:.0f}s")
        total += n
    print(f"raw.studies now has {con.execute('select count(*) from raw.studies').fetchone()[0]:,} trials" if total or done else "nothing to load")
    con.close()
    return total


if __name__ == "__main__":
    load_all()
