"""Evaluate the analyst agent on questions whose answers are computed directly with SQL.

Question families (gold answer from the warehouse, so the check is exact):
  count     "How many Phase 2 Neoplasms trials that started in 2014 were terminated?"
  rate      "What share of resolved Infections trials that started 2010-2015 were terminated?"  (+-1 pt)
  top_risk  "Which active Phase 3 trial in X has the highest termination risk?"               (NCT match)
  lookup    "What is the model's termination risk for NCT0...?"                               (+-0.01)
Also: citation validity (no NCT IDs the tools never returned), tool calls, latency, cost.

    python -m trialscope.agent_eval --n 24
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from trialscope import config, db
from trialscope.agent import NCT_RE, ask, build_agent

NUM_RE = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%)?")


def make_questions(n: int, seed: int = config.SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    t = db.query("select * from marts.fct_trials where start_year between 2008 and 2019")
    qs = []
    combos = t[t.status_group == "terminated"].groupby(["phase_group", "therapeutic_area", "start_year"]).size()
    combos = combos[combos >= 3].sample(frac=1, random_state=seed)
    for (ph, ar, yr), k in combos.head(n // 4).items():
        qs.append({"family": "count", "gold": int(k),
                   "question": f"How many {ph} trials in the therapeutic area '{ar}' that started in {yr} were terminated?"})
    areas = t.therapeutic_area.value_counts().index[:8]
    for ar in rng.choice(areas, size=min(n // 4, len(areas)), replace=False):
        d = t[(t.therapeutic_area == ar) & t.is_resolved & t.start_year.between(2010, 2015)]
        qs.append({"family": "rate", "gold": round(float(d.terminated_flag.mean()) * 100, 1),
                   "question": f"Among resolved (completed or terminated) '{ar}' trials that started 2010-2015, "
                               f"what percentage were terminated?"})
    r = db.query("select t.nct_id, t.phase_group, t.therapeutic_area, r.risk_termination, r.risk_rank_score "
                 "from marts.fct_trials t join ml.risk_scores r using (nct_id) where t.status_group = 'active'")
    groups = r.groupby(["phase_group", "therapeutic_area"]).size()
    for ph, ar in groups[groups >= 5].sample(frac=1, random_state=seed).index[: n // 4]:
        g = r[(r.phase_group == ph) & (r.therapeutic_area == ar)].sort_values("risk_rank_score", ascending=False)
        qs.append({"family": "top_risk", "gold": g.nct_id.iloc[0],
                   "question": f"Which active {ph} trial in '{ar}' has the highest predicted termination risk?"})
    for _, row in r.sample(n=min(n - len(qs), len(r)), random_state=seed).iterrows():
        qs.append({"family": "lookup", "gold": round(float(row.risk_termination), 3),
                   "question": f"What is the model's predicted termination risk for {row.nct_id}? Give the probability."})
    return qs


def numbers(text: str) -> list[float]:
    return [float(a.replace(",", "")) for a, _ in NUM_RE.findall(text)]


def correct(q: dict, answer: str) -> bool:
    nums = numbers(answer)
    if q["family"] == "count":
        return q["gold"] in [int(x) for x in nums if x == int(x)]
    if q["family"] == "rate":
        return any(abs(x - q["gold"]) <= 1.0 or abs(x * 100 - q["gold"]) <= 1.0 for x in nums)
    if q["family"] == "top_risk":
        ids = [c.upper() for c in NCT_RE.findall(answer)]
        return bool(ids) and ids[0] == q["gold"]
    if q["family"] == "lookup":
        return any(abs(x - q["gold"]) <= 0.011 or abs(x / 100 - q["gold"]) <= 0.011 for x in nums)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    qs = make_questions(a.n)
    app = build_agent()
    rows = []
    for q in qs:
        try:
            r = ask(q["question"], app=app)
        except Exception as e:
            r = {"answer": f"ERROR {e}", "tool_calls": [], "uncited_ids": [], "cited_trials": [], "latency_s": None,
                 "usage": {"cost_usd": 0, "llm_calls": 0}}
        rows.append({**q, "answer": r["answer"], "correct": correct(q, r["answer"]), "n_tool_calls": len(r["tool_calls"]),
                     "tools": [t["name"] for t in r["tool_calls"]], "unverified_ids": r["uncited_ids"],
                     "latency_s": r["latency_s"], "cost_usd": r["usage"]["cost_usd"]})
        print(f"[{'OK ' if rows[-1]['correct'] else 'BAD'}] {q['family']:8} gold={q['gold']}  {r['answer'][:90]!r}")
    df = pd.DataFrame(rows)
    summary = {"n": len(df), "accuracy": round(float(df.correct.mean()), 3),
               "accuracy_by_family": df.groupby("family").correct.mean().round(3).to_dict(),
               "answers_with_unverified_ids": int((df.unverified_ids.str.len() > 0).sum()),
               "mean_tool_calls": round(float(df.n_tool_calls.mean()), 2),
               "p50_latency_s": float(df.latency_s.median()), "cost_per_question_usd": round(float(df.cost_usd.mean()), 5)}
    df.to_json(config.RESULTS_DIR / "agent_eval.jsonl", orient="records", lines=True)
    (config.RESULTS_DIR / "agent_eval.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
