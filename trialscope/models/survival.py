"""Survival analysis of time from start to primary completion.

- Kaplan-Meier curves by phase and therapeutic area (+ median time to completion).
- Cox proportional-hazards model (lifelines) on start-time covariates -> hazard ratios.
- Gradient-boosted Cox model (XGBoost survival:cox) on the full leakage-safe feature set.
- Out-of-time concordance (C-index) vs a phase-only baseline.
- For active trials: P(primary completion later than currently planned) and expected slip.

Event = reached primary completion. Terminated trials are censored at their stop date and active
trials at the snapshot date (a competing-risks simplification; see README).

    python -m trialscope.models.survival
"""
from __future__ import annotations

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from trialscope import config
from trialscope.features import FeatureBuilder, load_frame

FIG = config.RESULTS_DIR / "figures"
COX_CATS = ["phase_group", "sponsor_class", "therapeutic_area"]


def survival_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.duration_days.notna() & (df.duration_days > 0) & df.status_group.isin(["completed", "terminated", "active"])].copy()
    d["T"] = d.duration_days / 30.44          # months
    d["E"] = d.completion_event.astype(int)
    return d


def cox_design(d: pd.DataFrame, levels: dict | None = None) -> tuple[pd.DataFrame, dict]:
    rare = d.is_rare_disease.astype(float).fillna(0)
    levels = levels or {**{c: d[c].value_counts().index[:8].tolist() for c in COX_CATS},
                        "_use_rare": bool(rare.std() > 0)}      # a constant column makes the Cox matrix singular
    X = pd.DataFrame(index=d.index)
    for c, lv in levels.items():
        if c.startswith("_"):
            continue
        for l in lv[1:]:                      # first level = reference
            X[f"{c}={l}"] = (d[c] == l).astype(float)
    if levels.get("_use_rare", True):
        X["is_rare_disease"] = rare
    X["log_n_exclusion"] = np.log1p(d.n_exclusion.fillna(0))
    X["requires_biomarker"] = d.requires_biomarker.astype(float).fillna(0)
    X["sponsor_prior_term_rate"] = d.sponsor_prior_term_rate.fillna(0.15)
    X["log_sponsor_prior_started"] = np.log1p(d.sponsor_prior_started.fillna(0))
    return X, levels


def km_plots(d: pd.DataFrame) -> pd.DataFrame:
    FIG.mkdir(exist_ok=True)
    rows = []
    for col, fname in (("phase_group", "km_by_phase.png"), ("therapeutic_area", "km_by_area.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for level in d[col].value_counts().index[:7]:
            g = d[d[col] == level]
            km = KaplanMeierFitter().fit(g["T"], g["E"], label=f"{level} (n={len(g):,})")
            km.plot_survival_function(ax=ax, ci_show=False)
            rows.append({"group": col, "level": level, "n": len(g), "median_months_to_completion": km.median_survival_time_})
        ax.set_xlabel("Months since start")
        ax.set_ylabel("Share not yet at primary completion")
        ax.set_xlim(0, 120)
        ax.set_title(f"Time to primary completion by {col.replace('_', ' ')}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=130)
        plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_frame()
    d = survival_frame(df)
    tr = d[(d.start_year >= config.MODEL_MIN_START_YEAR) & (d.start_year <= config.TRAIN_END_YEAR + 1)]   # train + val years
    te = d[(d.start_year >= config.TEST_START_YEAR) & (d.start_year <= config.TEST_END_YEAR)]
    km = km_plots(d[(d.start_year >= config.MODEL_MIN_START_YEAR) & (d.start_year <= config.TEST_END_YEAR)])

    # --- Cox PH (interpretable)
    Xtr, levels = cox_design(tr)
    Xte, _ = cox_design(te, levels)
    cph = CoxPHFitter(penalizer=0.01).fit(pd.concat([Xtr, tr[["T", "E"]]], axis=1), "T", "E")
    c_cox = concordance_index(te["T"], cph.predict_partial_hazard(Xte).pipe(lambda h: -h), te["E"])

    # --- phase-only baseline
    ph_cols = [c for c in Xtr.columns if c.startswith("phase_group=")]
    base = CoxPHFitter(penalizer=0.01).fit(pd.concat([Xtr[ph_cols], tr[["T", "E"]]], axis=1), "T", "E")
    c_base = concordance_index(te["T"], -base.predict_partial_hazard(Xte[ph_cols]), te["E"])

    # --- gradient-boosted Cox on the full leakage-safe feature set
    fb = FeatureBuilder().fit(tr)
    lab = np.where(tr["E"] == 1, tr["T"], -tr["T"])
    gb = xgb.XGBRegressor(objective="survival:cox", n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                          colsample_bytree=0.6, min_child_weight=10, random_state=config.SEED)
    gb.fit(fb.transform(tr), lab)
    c_gb = concordance_index(te["T"], -gb.predict(fb.transform(te)), te["E"])

    hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
        columns={"exp(coef)": "hazard_ratio", "exp(coef) lower 95%": "hr_lo", "exp(coef) upper 95%": "hr_hi"})
    res = {"c_index_phase_only": round(float(c_base), 4), "c_index_cox": round(float(c_cox), 4),
           "c_index_xgb_cox": round(float(c_gb), 4), "n_train": int(len(tr)), "n_test": int(len(te)),
           "test_event_rate": round(float(te["E"].mean()), 4)}
    with (config.MODEL_DIR / "survival.pkl").open("wb") as f:
        pickle.dump({"cox": cph, "levels": levels}, f)
    (config.RESULTS_DIR / "survival_metrics.json").write_text(json.dumps(res, indent=2))
    lines = ["# Survival analysis: time to primary completion", "",
             f"Out-of-time concordance (test start years {config.TEST_START_YEAR}-{config.TEST_END_YEAR}, n={len(te):,}):", "",
             f"| Model | C-index |", "|---|---|", f"| Phase only (Cox) | {c_base:.3f} |", f"| Cox PH, start-time covariates | {c_cox:.3f} |",
             f"| Gradient-boosted Cox (XGBoost), all features | {c_gb:.3f} |", "",
             "Hazard ratio > 1 = reaches primary completion FASTER; < 1 = slower.", "",
             hr.round(3).to_markdown(), "", "## Median months to primary completion (Kaplan-Meier)", "",
             km.round(1).to_markdown(index=False), "", "![KM by phase](figures/km_by_phase.png)", "![KM by area](figures/km_by_area.png)"]
    (config.RESULTS_DIR / "survival_report.md").write_text("\n".join(lines) + "\n")

    import mlflow          # imported here so the dashboard can import this module without mlflow installed

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("trialscope-survival")
    with mlflow.start_run(run_name="cox_vs_xgbcox"):
        mlflow.log_metrics(res)
        mlflow.log_artifact(str(config.RESULTS_DIR / "survival_report.md"))
    print("\n".join(lines[:9]))


def p_late(cph: CoxPHFitter, X: pd.DataFrame, elapsed_m: np.ndarray, planned_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(not completed by the planned date | still running now) and predicted median completion (months)."""
    sf = cph.predict_survival_function(X, times=np.arange(0, 241, 1.0))      # monthly grid, 20 years
    grid = sf.index.to_numpy()
    S = sf.to_numpy().T                                                     # (n, t)
    idx = lambda m: np.clip(np.searchsorted(grid, m), 0, len(grid) - 1)
    rows = np.arange(len(X))
    s_now = np.maximum(S[rows, idx(elapsed_m)], 1e-6)
    s_plan = S[rows, idx(np.maximum(planned_m, elapsed_m))]
    p = np.clip(s_plan / s_now, 0, 1)
    cond = S / s_now[:, None]
    cond[grid[None, :] < elapsed_m[:, None]] = 1.0
    median = np.array([grid[np.argmax(c <= 0.5)] if (c <= 0.5).any() else np.nan for c in cond])
    return p, median


if __name__ == "__main__":
    main()
