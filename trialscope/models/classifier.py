"""Termination-risk and accrual-failure classifiers with an out-of-time test.

Train: trials started <= TRAIN_END_YEAR; validation: VAL_YEAR (early stopping + Platt
calibration); test: TEST_START_YEAR..TEST_END_YEAR. Compared against simple baselines, plus a
leakage ablation that adds post-start fields to show how much they inflate the metrics.

    python -m trialscope.models.classifier
"""
from __future__ import annotations

import json
import pickle

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trialscope import config
from trialscope.models.calibration import PlattCalibrator
from trialscope.features import FeatureBuilder, load_frame, resolution_by_year, split

PARAMS = dict(n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=10, subsample=0.8,
              colsample_bytree=0.6, reg_lambda=2.0, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=100)


def flagged_at_launch(y: np.ndarray, p: np.ndarray, top: float) -> float:
    """Share of trials that eventually terminated that were in the model's top `top` fraction of risk."""
    cut = np.quantile(p, 1 - top)
    return float(((p >= cut) & (y == 1)).sum() / max((y == 1).sum(), 1))


def metrics(y, p, prefix: str, p_cal=None) -> dict:
    """Ranking metrics on raw scores (isotonic calibration creates ties); Brier on calibrated probabilities."""
    return {f"{prefix}auroc": roc_auc_score(y, p), f"{prefix}auprc": average_precision_score(y, p),
            f"{prefix}brier": brier_score_loss(y, np.clip(p if p_cal is None else p_cal, 0, 1)),
            f"{prefix}flagged_top10": flagged_at_launch(y, p, 0.10), f"{prefix}flagged_top20": flagged_at_launch(y, p, 0.20),
            f"{prefix}base_rate": float(np.mean(y))}


def fit_xgb(Xtr, ytr, Xva, yva):
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    m = xgb.XGBClassifier(**PARAMS, scale_pos_weight=spw, random_state=config.SEED, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def train_target(df: pd.DataFrame, target: str, leaky: bool = False):
    d = split(df)
    d = {k: v[v[target].notna()] for k, v in d.items()}
    fb = FeatureBuilder(leaky=leaky).fit(d["train"])
    X = {k: fb.transform(v) for k, v in d.items()}
    y = {k: v[target].astype(int).to_numpy() for k, v in d.items()}
    model = fit_xgb(X["train"], y["train"], X["val"], y["val"])
    raw = {k: model.predict_proba(X[k])[:, 1] for k in X}
    iso = PlattCalibrator().fit(raw["val"], y["val"])
    return d, fb, X, y, model, iso, raw


def baselines(d: dict, target: str) -> dict:
    tr, te = d["train"], d["test"]
    y = te[target].astype(int).to_numpy()
    out = {}
    # historical rate for the trial's phase x therapeutic area, learned on the training cohort
    rate = tr.groupby(["phase_group", "therapeutic_area"])[target].mean()
    p = np.array([rate.get((a, b), tr[target].mean()) for a, b in zip(te.phase_group, te.therapeutic_area)])
    out["phase_x_area_rate"] = metrics(y, p, "")
    # logistic regression on a handful of structured fields
    cols = ["phase_group", "sponsor_class", "therapeutic_area"]
    fb = FeatureBuilder(embeddings=False).fit(tr)
    keep = [c for c in fb.columns if c.split("=")[0] in cols] + ["sponsor_prior_term_rate", "is_rare_disease"]
    Xtr, Xte = fb.transform(tr)[keep].fillna(0), fb.transform(te)[keep].fillna(0)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(Xtr, tr[target].astype(int))
    out["logistic_structured"] = metrics(y, lr.predict_proba(Xte)[:, 1], "")
    return out


def global_drivers(model, X: pd.DataFrame) -> pd.DataFrame:
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=X.columns)
    return imp.groupby(FeatureBuilder.group_of).sum().sort_values(ascending=False).rename("mean_abs_shap").to_frame()


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for k successes out of n."""
    p = k / np.maximum(n, 1)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 8) -> pd.DataFrame:
    """Predicted vs observed termination rate per risk bin (out-of-time test cohort), with Wilson intervals."""
    b = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    g = pd.DataFrame({"p": p, "y": y, "bin": b.to_numpy()}).groupby("bin")
    t = g.agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"), k=("y", "sum"))
    t["lo"], t["hi"] = wilson(t.k.to_numpy(float), t.n.to_numpy(float))
    return t.drop(columns="k")


def plot_calibration(cal: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    top = float(max(cal.hi.max(), cal.predicted.max())) * 1.1
    ax.plot([0, top], [0, top], "--", color="#9aa5b1", label="perfect calibration")
    ax.errorbar(cal.predicted, cal.observed, yerr=[cal.observed - cal.lo, cal.hi - cal.observed], fmt="o-", color="#2f6fed",
                capsize=3, label="observed (95% CI)")
    ax.set_xlabel("predicted termination risk")
    ax.set_ylabel("observed termination rate")
    ax.set_title("Calibration on the out-of-time test cohort")
    ax.legend(frameon=False)
    fig.tight_layout()
    (config.RESULTS_DIR / "figures").mkdir(exist_ok=True)
    fig.savefig(config.RESULTS_DIR / "figures" / "calibration.png", dpi=140)
    plt.close(fig)


def slice_table(te: pd.DataFrame, y: np.ndarray, p: np.ndarray, col: str, top: int = 6) -> pd.DataFrame:
    rows = []
    for level in te[col].fillna("NA").value_counts().head(top).index:
        m = (te[col].fillna("NA") == level).to_numpy()
        if m.sum() >= 100 and 10 <= y[m].sum() <= m.sum() - 10:
            rows.append({col: level, "n": int(m.sum()), "termination_rate": round(float(y[m].mean()), 3),
                         "auroc": round(float(roc_auc_score(y[m], p[m])), 3)})
    return pd.DataFrame(rows)


def conformal_summary(yva, rva, yte, rte, alpha: float = 0.10) -> dict:
    """Split conformal prediction sets. Platt calibrator fit on half the validation year, conformal quantile on the other half."""
    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(yva))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    iso = PlattCalibrator().fit(rva[a], yva[a])
    pb, pt = iso.predict(rva[b]), iso.predict(rte)
    score = np.where(yva[b] == 1, 1 - pb, pb)
    qhat = np.quantile(score, min(1.0, np.ceil((len(b) + 1) * (1 - alpha)) / len(b)), method="higher")
    in1, in0 = (1 - pt) <= qhat, pt <= qhat
    covered = np.where(yte == 1, in1, in0)
    return {"alpha": alpha, "coverage": float(covered.mean()), "both": float((in1 & in0).mean()),
            "only_terminated": float((in1 & ~in0).mean()), "only_completed": float((~in1 & in0).mean()),
            "empty": float((~in1 & ~in0).mean())}


def extra_evaluation(d: dict, y: dict, raw: dict, iso) -> str:
    """Calibration, slices, per-year backtest and conformal coverage, as markdown for the report."""
    te, p_cal = d["test"], iso.predict(raw["test"])
    cal = calibration_table(y["test"], p_cal)
    plot_calibration(cal)
    out = ["## Calibration (out-of-time test cohort)", "",
           "A score of 0.30 should mean roughly 30% of such trials terminated. Observed rate per risk bin, with 95% Wilson intervals:", "",
           cal.round(3).to_markdown(index=False), "", "![calibration](figures/calibration.png)", "",
           "## Backtest: each test year scored by a model that never saw it", "",
           "| Start year | n | termination rate | AUROC | terminated trials caught in top 10% |", "|---|---|---|---|---|"]
    for yr in sorted(te.start_year.unique()):
        m = (te.start_year == yr).to_numpy()
        if m.sum() > 50 and 0 < y["test"][m].sum() < m.sum():
            out.append(f"| {int(yr)} | {int(m.sum())} | {y['test'][m].mean():.1%} | {roc_auc_score(y['test'][m], raw['test'][m]):.3f} | "
                       f"{flagged_at_launch(y['test'][m], raw['test'][m], 0.10):.1%} |")
    out += ["", "## Performance by slice (where is the model weaker?)", ""]
    for col in ("sponsor_class", "phase_group", "therapeutic_area"):
        t = slice_table(te, y["test"], raw["test"], col)
        if len(t):
            out += [t.to_markdown(index=False), ""]
    cf = conformal_summary(y["val"], raw["val"], y["test"], raw["test"])
    out += ["## Uncertainty: split-conformal prediction sets", "",
            f"Targeting {1 - cf['alpha']:.0%} coverage. Empirical coverage on the test cohort: **{cf['coverage']:.1%}**. "
            f"The model commits to one label for {cf['only_terminated'] + cf['only_completed']:.1%} of trials "
            f"({cf['only_terminated']:.1%} 'will terminate', {cf['only_completed']:.1%} 'will complete') and abstains "
            f"({cf['both']:.1%} 'could be either', {cf['empty']:.1%} empty) on the rest. "
            + ("With a ~10% base rate and start-time information only, a 90%-coverage set almost always contains just 'completed': "
             "the guarantee holds, but the sets are not informative here. The calibrated probability and the ranking carry the signal."
             if cf["only_completed"] > 0.9 else
             "The model abstains on the trials where start-time information is genuinely ambiguous."), ""]
    return "\n".join(out)


def main() -> None:
    df = load_frame()
    res_year = resolution_by_year(df[df.start_year >= config.MODEL_MIN_START_YEAR])
    report: dict = {"cohorts": {}, "termination": {}, "accrual_failure": {}}

    d, fb, X, y, model, iso, raw = train_target(df, "terminated_flag")
    for k in d:
        report["cohorts"][k] = {"n": int(len(d[k])), "years": f"{int(d[k].start_year.min())}-{int(d[k].start_year.max())}",
                                "termination_rate": round(float(y[k].mean()), 4)}
    report["termination"]["xgboost"] = metrics(y["test"], raw["test"], "", iso.predict(raw["test"]))
    report["termination"].update(baselines(d, "terminated_flag"))
    report["termination"]["xgboost_no_text"] = metrics(y["test"], *_no_text(df))
    # leakage ablation
    dl, fbl, Xl, yl, ml, isol, rawl = train_target(df, "terminated_flag", leaky=True)
    report["termination"]["xgboost_WITH_LEAKY_FEATURES"] = metrics(yl["test"], rawl["test"], "")
    drivers = global_drivers(model, X["test"])
    leak_drivers = global_drivers(ml, Xl["test"]).head(5)

    # accrual-failure model (terminated for low enrollment vs completed)
    da, fba, Xa, ya, ma, isoa, rawa = train_target(df, "accrual_failure")
    report["accrual_failure"]["xgboost"] = metrics(ya["test"], rawa["test"], "", isoa.predict(rawa["test"]))
    report["accrual_failure"].update(baselines(da, "accrual_failure"))

    report = json.loads(json.dumps(report, default=float), parse_float=lambda s: round(float(s), 4))
    with (config.MODEL_DIR / "termination.pkl").open("wb") as f:
        pickle.dump({"model": model, "calibrator": iso, "features": fb}, f)
    with (config.MODEL_DIR / "accrual.pkl").open("wb") as f:
        pickle.dump({"model": ma, "calibrator": isoa, "features": fba}, f)
    drivers.to_csv(config.RESULTS_DIR / "termination_drivers.csv")
    (config.RESULTS_DIR / "classifier_metrics.json").write_text(json.dumps(report, indent=2))
    write_report(report, drivers, leak_drivers, res_year, extra_evaluation(d, y, raw, iso))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("trialscope-termination")
    with mlflow.start_run(run_name="xgb_termination_oot"):
        mlflow.log_params({k: v for k, v in PARAMS.items()} | {"train_end": config.TRAIN_END_YEAR, "val_year": config.VAL_YEAR,
                          "test": f"{config.TEST_START_YEAR}-{config.TEST_END_YEAR}", "n_features": len(fb.columns),
                          "embed_model": config.EMBED_MODEL, "best_iteration": model.best_iteration})
        for target in ("termination", "accrual_failure"):
            for name, m in report[target].items():
                mlflow.log_metrics({f"{target}.{name}.{k}": v for k, v in m.items()})
        mlflow.log_artifact(str(config.RESULTS_DIR / "classifier_report.md"))
    print((config.RESULTS_DIR / "classifier_report.md").read_text())


def _no_text(df):
    """Same model without text features — how much do eligibility criteria add?"""
    d = split(df)
    fb = FeatureBuilder(embeddings=False).fit(d["train"])
    drop = ["n_inclusion", "n_exclusion", "elig_chars", "ecog_max", "n_lab_thresholds", "requires_biomarker",
            "requires_prior_therapy", "excludes_brain_mets"]
    X = {k: fb.transform(v).drop(columns=drop) for k, v in d.items()}
    y = {k: v.terminated_flag.astype(int).to_numpy() for k, v in d.items()}
    m = fit_xgb(X["train"], y["train"], X["val"], y["val"])
    iso = PlattCalibrator().fit(m.predict_proba(X["val"])[:, 1], y["val"])
    raw = m.predict_proba(X["test"])[:, 1]
    return raw, "", iso.predict(raw)


def write_report(r: dict, drivers: pd.DataFrame, leak_drivers: pd.DataFrame, res_year: pd.DataFrame, extra: str = "") -> None:
    def table(block: dict) -> str:
        rows = ["| Model | AUROC | AUPRC | Brier | Terminated trials flagged in top 10% | in top 20% |", "|---|---|---|---|---|---|"]
        for name, m in block.items():
            rows.append(f"| {name} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['brier']:.3f} | {m['flagged_top10']:.1%} | {m['flagged_top20']:.1%} |")
        return "\n".join(rows)

    c = r["cohorts"]
    text = [
        "# Out-of-time classifier report", "",
        f"Train {c['train']['years']} (n={c['train']['n']:,}), validation {c['val']['years']} (n={c['val']['n']:,}), "
        f"test {c['test']['years']} (n={c['test']['n']:,}, termination rate {c['test']['termination_rate']:.1%}). "
        "Only features known at each trial's start are used.", "",
        "## Termination (terminated vs completed)", "", table(r["termination"]), "",
        "`xgboost_WITH_LEAKY_FEATURES` adds post-start fields (actual enrollment, final site count, results posted...). "
        "Its inflated score is what a leaky pipeline would report; top drivers of that leaky model: "
        + ", ".join(f"`{i}`" for i in leak_drivers.index) + ".", "",
        "## Accrual failure (terminated for low enrollment vs completed)", "", table(r["accrual_failure"]), "",
        "## Top risk drivers (mean |SHAP|, test cohort)", "", drivers.head(15).round(4).to_markdown(), "",
        extra, "",
        "## Why recent start years are not used for testing", "",
        "Trials that started recently and already ended are disproportionately the ones that stopped early, so "
        "evaluating on them would be biased. Share of trials resolved by start year:", "",
        res_year.tail(12).to_markdown(),
    ]
    (config.RESULTS_DIR / "classifier_report.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
