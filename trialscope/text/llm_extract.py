"""LLM structured extraction from eligibility criteria, evaluated against hand labels.

The LLM reads the messy criteria text and returns typed fields. Its accuracy (and the rule
baseline's) is measured on labels/eligibility_labels.jsonl, which you create with
`python -m trialscope.text.label eligibility --n 50` (labelling is blind to model output).

    python -m trialscope.text.llm_extract run --n 300      # extract for labelled trials + a random sample
    python -m trialscope.text.llm_extract eval             # field accuracy: rules vs LLM
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm

from trialscope import config, db
from trialscope.llm import UsageTracker, get_chat
from trialscope.text.eligibility import rule_features

FIELDS = ["n_inclusion", "n_exclusion", "requires_biomarker", "requires_prior_therapy", "ecog_max", "excludes_brain_mets"]
COUNT_FIELDS = {"n_inclusion", "n_exclusion"}


class EligibilityExtraction(BaseModel):
    n_inclusion: int = Field(description="number of distinct inclusion criteria")
    n_exclusion: int = Field(description="number of distinct exclusion criteria")
    requires_biomarker: bool = Field(description="inclusion requires a molecular/genetic/protein biomarker (e.g. EGFR mutation, HER2+, PD-L1)")
    biomarkers: list[str] = Field(default_factory=list, description="biomarkers required for inclusion")
    requires_prior_therapy: bool = Field(description="inclusion requires prior treatment / relapsed or refractory disease")
    ecog_max: Optional[int] = Field(None, description="highest ECOG performance status allowed, null if not stated")
    excludes_brain_mets: bool = Field(description="brain/CNS metastases are an exclusion")


PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract structured fields from clinical-trial eligibility criteria. Count each distinct criterion once "
               "(sub-bullets of one criterion count as one). Only report what the text states."),
    ("human", "{text}"),
])


def run(n: int, workers: int = 8) -> pd.DataFrame:
    labelled = _labels().nct_id.tolist() if (config.LABELS_DIR / "eligibility_labels.jsonl").exists() else []
    done = set(db.query("select nct_id from ml.llm_eligibility").nct_id) if db.table_exists("llm_eligibility") else set()
    pool = db.query("select nct_id, eligibility_criteria from staging.stg_studies where length(eligibility_criteria) > 50")
    pool = pool.set_index("nct_id")
    sample = [x for x in labelled if x in pool.index] + \
        pool.drop(index=[x for x in labelled if x in pool.index]).sample(n=min(n, len(pool)), random_state=config.SEED).index.tolist()
    todo = [x for x in dict.fromkeys(sample) if x not in done][: max(n, len(labelled))]
    chain = PROMPT | get_chat().with_structured_output(EligibilityExtraction, method="function_calling")
    tracker, rows = UsageTracker(), []

    def work(nct):
        r = chain.invoke({"text": pool.at[nct, "eligibility_criteria"][:12000]}, {"callbacks": [tracker]})
        return {"nct_id": nct, **r.model_dump(), "biomarkers": json.dumps(r.biomarkers)}

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(work, x): x for x in todo}
        for f in tqdm(as_completed(futs), total=len(futs), desc="LLM extraction"):
            try:
                rows.append(f.result())
            except Exception as e:
                tqdm.write(f"  ! {futs[f]}: {e}")
    new = pd.DataFrame(rows)
    old = db.query("select * from ml.llm_eligibility") if done else pd.DataFrame()
    out = pd.concat([old, new], ignore_index=True)
    if len(out):
        db.write_table(out, "llm_eligibility")
    u = tracker.summary()
    print(f"extracted {len(new)} new ({len(out)} total); cost ${u['cost_usd']:.3f}, "
          f"${u['cost_usd'] / max(len(new), 1) * 1000:.2f} per 1,000 trials")
    return out


def _labels() -> pd.DataFrame:
    return pd.read_json(config.LABELS_DIR / "eligibility_labels.jsonl", lines=True)


def _agree(field: str, pred, gold) -> bool:
    if pd.isna(gold) and (pred is None or pd.isna(pred)):
        return True
    if pd.isna(gold) or pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return False
    return int(pred) == int(gold) if field in COUNT_FIELDS or field == "ecog_max" else bool(pred) == bool(gold)


def evaluate() -> dict:
    gold = _labels()
    text = db.query("select nct_id, eligibility_criteria from staging.stg_studies").set_index("nct_id").eligibility_criteria
    llm = db.query("select * from ml.llm_eligibility").set_index("nct_id") if db.table_exists("llm_eligibility") else pd.DataFrame()
    res = {"n_labelled": len(gold), "fields": {}}
    for f in FIELDS:
        row = {}
        for name, getter in (("rules", lambda n: rule_features(text.get(n))[f]),
                             ("llm", lambda n: llm.at[n, f] if n in llm.index else None)):
            preds = [(getter(n), g) for n, g in zip(gold.nct_id, gold[f]) if name == "rules" or n in llm.index]
            if not preds:
                continue
            row[f"{name}_acc"] = round(sum(_agree(f, p, g) for p, g in preds) / len(preds), 3)
            if f in COUNT_FIELDS:
                row[f"{name}_within1"] = round(sum(p is not None and abs(int(p) - int(g)) <= 1 for p, g in preds) / len(preds), 3)
        res["fields"][f] = row
    for name in ("rules", "llm"):
        accs = [v[f"{name}_acc"] for v in res["fields"].values() if f"{name}_acc" in v]
        if accs:
            res[f"{name}_mean_acc"] = round(sum(accs) / len(accs), 3)
    (config.RESULTS_DIR / "extraction_eval.json").write_text(json.dumps(res, indent=2))
    table = pd.DataFrame(res["fields"]).T
    (config.RESULTS_DIR / "extraction_eval.md").write_text(
        f"# Eligibility extraction accuracy\n\n{len(gold)} hand-labelled trials, chat model `{config.CHAT_MODEL}`.\n\n"
        + table.to_markdown() + f"\n\nMean field accuracy: rules {res.get('rules_mean_acc')}, LLM {res.get('llm_mean_acc')}\n")
    print(table.to_string())
    print({k: v for k, v in res.items() if k != "fields"})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "eval"])
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    run(a.n) if a.cmd == "run" else evaluate()
