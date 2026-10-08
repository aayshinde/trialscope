"""Tiny terminal labelling tool for the hand-labelled evaluation sets.

You see only the source text (never a model's answer), so labels aren't anchored on predictions.

    python -m trialscope.text.label eligibility --n 50     # -> labels/eligibility_labels.jsonl
    python -m trialscope.text.label reasons --n 100        # -> labels/reason_labels.csv

Answer each prompt and press Enter. Type s to skip a trial, q to save and quit.
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap

from trialscope import config, db
from trialscope.text.termination import CATEGORIES

ELIG_PATH = config.LABELS_DIR / "eligibility_labels.jsonl"
REASON_PATH = config.LABELS_DIR / "reason_labels.csv"


class Quit(Exception):
    pass


class Skip(Exception):
    pass


def ask(prompt: str, kind: str):
    while True:
        a = input(f"  {prompt}: ").strip().lower()
        if a == "q":
            raise Quit
        if a == "s":
            raise Skip
        if kind == "int" and a.isdigit():
            return int(a)
        if kind == "int?" and (a.isdigit() or a in ("", "none", "-")):
            return int(a) if a.isdigit() else None
        if kind == "bool" and a in ("y", "n", "yes", "no"):
            return a.startswith("y")
        print("   (int / y,n / empty for none; s=skip, q=quit)")


def label_eligibility(n: int) -> None:
    done = {json.loads(l)["nct_id"] for l in ELIG_PATH.read_text().splitlines()} if ELIG_PATH.exists() else set()
    df = db.query("select nct_id, eligibility_criteria from staging.stg_studies where length(eligibility_criteria) > 50 "
                  "order by hash(nct_id) limit ?", [n + len(done) + 50])
    df = df[~df.nct_id.isin(done)].head(n)
    with ELIG_PATH.open("a") as f:
        for i, r in enumerate(df.itertuples(), 1):
            print("\n" + "=" * 90 + f"\n[{i}/{len(df)}] {r.nct_id}\n" + "=" * 90)
            print("\n".join(textwrap.fill(l, 110) for l in r.eligibility_criteria.splitlines()))
            try:
                rec = {"nct_id": r.nct_id,
                       "n_inclusion": ask("number of inclusion criteria", "int"),
                       "n_exclusion": ask("number of exclusion criteria", "int"),
                       "requires_biomarker": ask("requires a biomarker for inclusion? (y/n)", "bool"),
                       "requires_prior_therapy": ask("requires prior therapy / relapsed-refractory? (y/n)", "bool"),
                       "ecog_max": ask("max ECOG allowed (empty if not stated)", "int?"),
                       "excludes_brain_mets": ask("brain metastases excluded? (y/n)", "bool")}
            except Skip:
                continue
            except Quit:
                break
            f.write(json.dumps(rec) + "\n")
            f.flush()
    print(f"saved to {ELIG_PATH}")


def label_reasons(n: int) -> None:
    done = set()
    if REASON_PATH.exists():
        with REASON_PATH.open() as f:
            done = {r["nct_id"] for r in csv.DictReader(f)}
    df = db.query("select nct_id, why_stopped from marts.fct_trials where why_stopped is not null order by hash(nct_id) limit ?",
                  [n + len(done) + 50])
    df = df[~df.nct_id.isin(done)].head(n)
    new = not REASON_PATH.exists()
    menu = "  ".join(f"{i}={c}" for i, c in enumerate(CATEGORIES))
    with REASON_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["nct_id", "label"])
        for i, r in enumerate(df.itertuples(), 1):
            print(f"\n[{i}/{len(df)}] {r.nct_id}: {r.why_stopped}\n  {menu}")
            try:
                k = ask("category number", "int")
            except Skip:
                continue
            except Quit:
                break
            if 0 <= k < len(CATEGORIES):
                w.writerow([r.nct_id, CATEGORIES[k]])
                f.flush()
    print(f"saved to {REASON_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["eligibility", "reasons"])
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    label_eligibility(a.n) if a.task == "eligibility" else label_reasons(a.n)
