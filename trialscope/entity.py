"""Sponsor entity resolution: "Pfizer", "Pfizer Inc.", "PFIZER, INC" -> one sponsor.

1. Rule-based key: lower-case, '&' -> 'and', drop punctuation and parenthesised acronyms,
   join initialisms (m d anderson -> md anderson), strip legal/corporate suffixes.
2. Curated aliases (labels/sponsor_aliases.csv) for cases rules can't know (Roche, UCSF, GSK).
3. Fuzzy merge of remaining keys with rapidfuzz (ratio >= 92, blocked on first token), so
   typos and spacing variants merge without collapsing different universities.
Writes raw.sponsor_map(raw_name, sponsor_key, sponsor_name) for dbt.

    python -m trialscope.entity            # build the map + print stats
    python -m trialscope.entity --eval     # pairwise precision/recall vs labels/sponsor_gold.csv
"""
from __future__ import annotations

import argparse
import itertools
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

from trialscope import config
from trialscope.ingest.load import connect

LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "gmbh", "ag", "sa", "sas",
         "plc", "bv", "nv", "kk", "spa", "srl", "lp", "pty", "ab", "as", "oy", "pharmaceuticals", "pharmaceutical",
         "pharma", "pharmaceutica", "the", "group", "holdings", "international", "usa", "us"}
FUZZY_THRESHOLD = 92


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\([^)]*\)", " ", s)                 # (NCI), (MSD)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = s.split()
    out, buf = [], ""
    for t in toks:                                    # m d anderson -> md anderson
        if len(t) == 1 and t.isalpha():
            buf += t
            continue
        if buf:
            out.append(buf)
            buf = ""
        out.append(t)
    if buf:
        out.append(buf)
    while out and out[-1] in LEGAL:                   # trailing legal suffixes only
        out.pop()
    out = [t for t in out if t not in ("the",)]
    return " ".join(out) or s.strip()


def load_aliases() -> dict[str, str]:
    p = config.LABELS_DIR / "sponsor_aliases.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {normalize(a): normalize(c) for a, c in zip(df.alias, df.canonical)}


def resolve(names: pd.Series, use_aliases: bool = True) -> pd.DataFrame:
    """names: raw sponsor name per trial (duplicates allowed, used as frequency)."""
    counts = names.dropna().value_counts()
    aliases = load_aliases() if use_aliases else {}
    keys = {n: aliases.get(normalize(n), normalize(n)) for n in counts.index}

    # fuzzy merge between distinct keys, blocked by first token to stay O(n * block)
    key_freq = pd.Series(counts.values, index=[keys[n] for n in counts.index]).groupby(level=0).sum().sort_values(ascending=False)
    parent = {k: k for k in key_freq.index}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    blocks: dict[str, list[str]] = {}
    for k in key_freq.index:
        blocks.setdefault(k.split()[0] if k else "", []).append(k)
    for block in blocks.values():
        if len(block) > 2000:          # very common first tokens (e.g. "university"): compare within first two tokens
            sub: dict[str, list[str]] = {}
            for k in block:
                sub.setdefault(" ".join(k.split()[:2]), []).append(k)
            groups = sub.values()
        else:
            groups = [block]
        for grp in groups:
            for a, b in itertools.combinations(grp, 2):     # grp is frequency-ordered: merge into the more frequent
                if abs(len(a) - len(b)) <= 6 and fuzz.ratio(a, b) >= FUZZY_THRESHOLD:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra

    final_key = {n: find(keys[n]) for n in counts.index}
    df = pd.DataFrame({"raw_name": counts.index, "n_trials": counts.values, "sponsor_key": [final_key[n] for n in counts.index]})
    # display name: most frequent variant, preferring ones that aren't SHOUTED and have no legal suffix
    df["_upper"] = df.raw_name.str.isupper()
    df["_suffix"] = df.raw_name.map(lambda n: normalize(n) != " ".join(re.sub(r"[^a-z0-9 ]+", " ", n.lower()).split()))
    canon = (df.sort_values(["_upper", "_suffix", "n_trials"], ascending=[True, True, False])
             .drop_duplicates("sponsor_key").set_index("sponsor_key").raw_name)
    df = df.drop(columns=["_upper", "_suffix"])
    df["sponsor_name"] = df.sponsor_key.map(canon)
    return df


def build() -> pd.DataFrame:
    con = connect()
    names = con.execute("select lead_sponsor_name from raw.studies").df().lead_sponsor_name
    df = resolve(names)
    con.register("m", df[["raw_name", "sponsor_key", "sponsor_name"]])
    con.execute("create or replace table raw.sponsor_map as select * from m")
    con.close()
    print(f"{df.raw_name.nunique():,} raw sponsor names -> {df.sponsor_key.nunique():,} sponsors "
          f"({1 - df.sponsor_key.nunique() / df.raw_name.nunique():.1%} reduction)")
    merged = df.groupby("sponsor_key").filter(lambda g: len(g) > 1).sort_values(["sponsor_key", "n_trials"], ascending=[True, False])
    print("Examples of merged variants:")
    for k, g in itertools.islice(merged.groupby("sponsor_key"), 8):
        print(f"  {g.sponsor_name.iloc[0]!r:45} <- {[n for n in g.raw_name if n != g.sponsor_name.iloc[0]][:3]}")
    return df


def evaluate(gold_path=None) -> dict:
    """Pairwise precision/recall over all name pairs in the gold file (columns: raw_name, entity).
    Reported with and without the curated alias table (aliases overlap the gold set, so the
    rules-only number is the honest measure of the automatic part)."""
    gold = pd.read_csv(gold_path or config.LABELS_DIR / "sponsor_gold.csv")
    out = {}
    for name, use in (("rules_only", False), ("rules_plus_aliases", True)):
        out[name] = _pairwise(gold, resolve(gold.raw_name, use).set_index("raw_name").sponsor_key)
    print(out)
    return out


def _pairwise(gold: pd.DataFrame, pred: pd.Series) -> dict:
    tp = fp = fn = 0
    for (a, ea), (b, eb) in itertools.combinations(zip(gold.raw_name, gold.entity), 2):
        same_pred, same_gold = pred[a] == pred[b], ea == eb
        tp += same_pred and same_gold
        fp += same_pred and not same_gold
        fn += same_gold and not same_pred
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"pairs_precision": round(p, 4), "pairs_recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "n_names": len(gold), "n_entities": gold.entity.nunique()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    evaluate() if a.eval else build()
