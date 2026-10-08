"""Weekly watchlist: the active trials the model considers most at risk, and what changed since the last run.

    python -m trialscope.watchlist          # writes results/watchlist.md and results/watchlist.csv
"""
from __future__ import annotations

import json

import pandas as pd

from trialscope import config, db

TOP_N = 15
CURRENT = config.RESULTS_DIR / "watchlist.csv"
PREVIOUS = config.RESULTS_DIR / "watchlist_previous.csv"


def build(n: int = TOP_N) -> pd.DataFrame:
    df = db.query("""select t.nct_id, t.brief_title, t.sponsor_name, t.phase_group, t.therapeutic_area, t.start_date,
                            r.risk_termination, r.risk_accrual, r.p_late, r.expected_slip_days, r.drivers
                     from marts.fct_trials t join ml.risk_scores r using (nct_id)
                     where t.status_group = 'active' and t.start_year >= year(t.snapshot_date) - 2
                     order by r.risk_rank_score desc limit ?""", [n])
    df["top_driver"] = df.drivers.map(lambda s: json.loads(s)[0]["driver"] if isinstance(s, str) and s else None)
    return df.drop(columns="drivers")


def render(df: pd.DataFrame, new_ids: set[str], snapshot: str) -> str:
    lines = [f"# TrialScope watchlist: {len(df)} highest-risk active trials", "",
             f"Active trials that started in the last two years, ranked by model termination risk. Data as of {snapshot}. "
             "Scores are model estimates, not facts.", "",
             "| # | Trial | Phase | Area | Sponsor | Termination risk | Accrual risk | Main driver | |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(df.itertuples(), 1):
        title = (r.brief_title or "")[:70]
        lines.append(f"| {i} | [{r.nct_id}](https://clinicaltrials.gov/study/{r.nct_id}) {title} | {r.phase_group} | {r.therapeutic_area} | "
                     f"{r.sponsor_name} | {r.risk_termination:.0%} | {r.risk_accrual:.0%} | {r.top_driver} | "
                     f"{'NEW' if r.nct_id in new_ids else ''} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = build()
    snap = str(db.query("select max(snapshot_date) d from marts.fct_trials").d[0])[:10]
    if CURRENT.exists():
        CURRENT.replace(PREVIOUS)
    previous = set(pd.read_csv(PREVIOUS).nct_id) if PREVIOUS.exists() else set()
    new_ids = set(df.nct_id) - previous if previous else set()
    df.to_csv(CURRENT, index=False)
    (config.RESULTS_DIR / "watchlist.md").write_text(render(df, new_ids, snap))
    print(f"watchlist: {len(df)} trials, {len(new_ids)} new since the previous run")


if __name__ == "__main__":
    main()
