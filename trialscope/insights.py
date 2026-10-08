"""Exploratory findings for the write-up: results/insights.md + results/figures/*.png.

Questions answered (resolved trials in the modelling window):
  1. Which therapeutic areas terminate most?            (with 95% Wilson intervals)
  2. Do trials with more sites recruit faster?           (total vs per-site enrollment rate)
  3. Does a sponsor's track record predict termination?  (5-year history, as of start)
  4. Do rare-disease trials struggle more?               (termination, accrual failure, duration)
  5. Why do trials stop, by phase?

    python -m trialscope.insights
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trialscope import config
from trialscope.features import load_frame

FIG = config.RESULTS_DIR / "figures"


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_table(d: pd.DataFrame, by: str, min_n: int = 30) -> pd.DataFrame:
    g = d.groupby(by).terminated_flag.agg(["sum", "count"])
    g = g[g["count"] >= min_n]
    ci = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    return pd.DataFrame({"n": g["count"], "termination_rate": g["sum"] / g["count"],
                         "ci_low": [c[0] for c in ci], "ci_high": [c[1] for c in ci]}).sort_values("termination_rate", ascending=False)


def barh(t: pd.DataFrame, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.2))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.termination_rate, color="#4C78A8")
    ax.errorbar(t.termination_rate, y, xerr=[t.termination_rate - t.ci_low, t.ci_high - t.termination_rate],
                fmt="none", ecolor="#333", capsize=2, lw=1)
    ax.set_yticks(y, [str(i)[:45] for i in t.index], fontsize=8)
    ax.set_xlabel("Share terminated (of completed + terminated)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    d = df[df.is_resolved.fillna(False) & (df.start_year >= config.MODEL_MIN_START_YEAR) & (df.start_year <= config.TEST_END_YEAR)].copy()
    out = ["# TrialScope findings", "",
           f"{len(d):,} resolved interventional trials started {config.MODEL_MIN_START_YEAR}-{config.TEST_END_YEAR} "
           f"(overall termination rate {d.terminated_flag.mean():.1%}).", ""]

    # 1. therapeutic areas
    ta = rate_table(d, "therapeutic_area")
    barh(ta, "Termination rate by therapeutic area (95% CI)", "termination_by_area.png")
    out += ["## 1. Termination by therapeutic area", "", ta.round(3).to_markdown(), "",
            f"Highest: **{ta.index[0]}** ({ta.termination_rate.iloc[0]:.1%}); lowest: **{ta.index[-1]}** "
            f"({ta.termination_rate.iloc[-1]:.1%}).", "", "![](figures/termination_by_area.png)", ""]
    ph = rate_table(d, ["phase_group", "sponsor_class"], min_n=50).round(3)
    out += ["### By phase x sponsor class", "", ph.to_markdown(), ""]

    # 2. sites vs recruitment speed (completed trials; uses final site counts — descriptive, NOT a model feature)
    c = d[(d.status_group == "completed") & (d.enrollment_type == "ACTUAL") & (d.duration_days > 30) & (d.n_locations > 0)].copy()
    c["months"] = c.duration_days / 30.44
    c["pts_per_month"] = c.enrollment_count / c.months
    c["pts_per_site_month"] = c.pts_per_month / c.n_locations
    c["sites"] = pd.cut(c.n_locations, [0, 1, 5, 20, 100, np.inf], labels=["1", "2-5", "6-20", "21-100", "100+"])
    st = c.groupby("sites", observed=True).agg(n=("nct_id", "size"), median_patients_per_month=("pts_per_month", "median"),
                                               median_patients_per_site_month=("pts_per_site_month", "median"),
                                               median_months=("months", "median")).round(2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    st.median_patients_per_month.plot.bar(ax=ax[0], color="#4C78A8", title="Patients enrolled per month (median)")
    st.median_patients_per_site_month.plot.bar(ax=ax[1], color="#F58518", title="Patients per site per month (median)")
    for a in ax:
        a.set_xlabel("Number of sites")
    fig.tight_layout()
    fig.savefig(FIG / "sites_vs_recruitment.png", dpi=130)
    plt.close(fig)
    out += ["## 2. Do more sites recruit faster?", "", st.to_markdown(), "",
            "Total enrollment speed vs per-site productivity. Final site counts are only known after the fact, so this is "
            "descriptive and the variable is excluded from the predictive model.", "", "![](figures/sites_vs_recruitment.png)", ""]

    # 3. sponsor track record
    s = d[d.sponsor_prior_closed >= 5].copy()
    if len(s) > 100:
        s["track_record"] = pd.qcut(s.sponsor_prior_term_rate, 5, duplicates="drop")
        tr = s.groupby("track_record", observed=True).terminated_flag.agg(n="size", termination_rate="mean").round(3)
        out += ["## 3. Sponsor track record (5 years before start, as of start date)", "",
                "Sponsors with at least 5 closed trials in the window, grouped into quintiles of their past termination rate:", "",
                tr.to_markdown(), ""]

    # 4. rare diseases
    r = d.groupby("is_rare_disease").agg(n=("nct_id", "size"), termination_rate=("terminated_flag", "mean"),
                                        accrual_failure_rate=("accrual_failure", "mean"))
    comp = d[d.completion_event == 1].groupby("is_rare_disease").duration_days.median() / 30.44
    r["median_months_to_completion"] = comp
    out += ["## 4. Rare-disease trials", "", r.round(3).to_markdown(), ""]

    # 5. reasons by phase
    t = df[(df.status_group == "terminated") & df.reason.notna()]
    if len(t):
        rp = pd.crosstab(t.phase_group, t.reason, normalize="index").round(3)
        out += ["## 5. Why trials stop, by phase (share of terminated trials)", "", rp.to_markdown(), ""]

    out += ["## Write-up prompt", "", "Pick the result that surprised you most, check it against the raw data "
            "(open a few trials on clinicaltrials.gov), and write it up with the figure."]
    (config.RESULTS_DIR / "insights.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:4]))
    print(f"wrote {config.RESULTS_DIR / 'insights.md'} and figures/")


if __name__ == "__main__":
    main()
