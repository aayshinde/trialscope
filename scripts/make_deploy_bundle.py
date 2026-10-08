"""Build the slim bundle a hosted copy of the dashboard runs on (Streamlit Community Cloud, Hugging Face Spaces...).

The full warehouse is ~180 MB and holds raw API payloads. The dashboard only needs the marts, the model outputs and, for the
what-if tool, the eligibility text of recent active trials. Output goes to deploy/ (committed to the repo); app.py falls back to
it automatically when no warehouse exists.

    python scripts/make_deploy_bundle.py [--text-trials 2500]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trialscope import config  # noqa: E402

OUT = ROOT / "deploy"
TABLES = ["marts.fct_trials", "marts.dim_sponsors", "marts.mart_area_phase_stats", "ml.risk_scores", "ml.similar_trials",
          "ml.termination_reasons", "ml.eligibility_features", "ml.text_embeddings", "ml.trial_drug_approval",
          "snapshots.snap_studies", "raw._loaded_files"]
# tables copied with only the columns the dashboard reads (snap_studies otherwise drags in every trial's full eligibility text)
COLUMNS = {"snapshots.snap_studies": "nct_id, dbt_valid_from, dbt_valid_to, overall_status, primary_completion_date, "
                                     "primary_completion_type, enrollment_count, enrollment_type"}
MODELS = ["termination.pkl", "accrual.pkl", "text_embedder.pkl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-trials", type=int, default=2500, help="how many recent active trials keep their eligibility text (what-if tool)")
    n_text = ap.parse_args().text_trials
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    dest = OUT / "trialscope_demo.duckdb"
    con = duckdb.connect(str(dest))
    con.execute(f"attach '{config.WAREHOUSE}' as trialscope (read_only)")
    existing = {(r[0], r[1]) for r in con.execute("select schema_name, table_name from duckdb_tables() where database_name = 'trialscope'").fetchall()}
    for full in TABLES:
        schema, table = full.split(".")
        if (schema, table) not in existing:
            continue
        con.execute(f"create schema if not exists {schema}")
        con.execute(f"create table {schema}.{table} as select {COLUMNS.get(full, '*')} from trialscope.{schema}.{table}")
    con.execute("create schema if not exists staging")
    con.execute(f"""create table staging.stg_studies as
                    select s.nct_id, s.brief_title, s.eligibility_criteria from trialscope.staging.stg_studies s
                    join trialscope.ml.risk_scores r using (nct_id) join trialscope.marts.fct_trials t using (nct_id)
                    where t.status_group = 'active' order by r.risk_rank_score desc limit {n_text}""")
    con.execute("detach trialscope")
    con.execute("checkpoint")
    con.close()
    for m in MODELS:
        shutil.copy(config.MODEL_DIR / m, OUT / "models" / m)
    shutil.copytree(config.RESULTS_DIR, OUT / "results", ignore=shutil.ignore_patterns("*.csv"))
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.stat().st_size > 1_000_000:
            print(f"{p.relative_to(OUT)}: {p.stat().st_size / 1e6:.1f} MB")
    print(f"deploy/ total: {sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
