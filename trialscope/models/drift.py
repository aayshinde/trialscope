"""Drift monitoring: are newly registered trials different from what the model was trained on?

Population Stability Index (PSI) per feature and for the risk score, comparing the training
cohort with trials that started in the last 24 months. PSI < 0.1 stable, 0.1-0.25 moderate,
> 0.25 significant shift. Run after every incremental load.

    python -m trialscope.models.drift
"""
from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd

from trialscope import config, db
from trialscope.features import CATEGORICAL, NUMERIC, load_frame


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) < 20 or len(cur) < 20:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    lv = set(ref.dropna()) | set(cur.dropna())
    r = ref.value_counts(normalize=True).reindex(lv, fill_value=0).clip(lower=1e-4)
    c = cur.value_counts(normalize=True).reindex(lv, fill_value=0).clip(lower=1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def main() -> pd.DataFrame:
    df = load_frame()
    snap = pd.Timestamp(df.snapshot_date.dropna().max())
    ref = df[(df.start_year >= config.MODEL_MIN_START_YEAR) & (df.start_year <= config.TRAIN_END_YEAR)]
    cur = df[pd.to_datetime(df.start_date) >= snap - pd.DateOffset(months=24)]
    rows = [{"feature": c, "type": "numeric", "psi": psi_numeric(ref[c].astype(float), cur[c].astype(float))}
            for c in NUMERIC if c in df]
    rows += [{"feature": c, "type": "categorical", "psi": psi_categorical(ref[c], cur[c])} for c in CATEGORICAL]
    if db.table_exists("risk_scores"):
        rs = db.query("select nct_id, risk_rank_score from ml.risk_scores")
        rows.append({"feature": "MODEL SCORE", "type": "prediction",
                     "psi": psi_numeric(ref.merge(rs).risk_rank_score, cur.merge(rs).risk_rank_score)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out["status"] = pd.cut(out.psi, [-1, 0.1, 0.25, np.inf], labels=["stable", "moderate", "SHIFT"])
    out["checked_at"] = pd.Timestamp.now()
    db.write_table(out, "drift_report")
    text = (f"# Drift report\n\nReference: trials started <= {config.TRAIN_END_YEAR} (n={len(ref):,}). "
            f"Current: trials started in the 24 months before {snap.date()} (n={len(cur):,}).\n\n"
            + out.drop(columns="checked_at").round(3).to_markdown(index=False) + "\n")
    (config.RESULTS_DIR / "drift.md").write_text(text)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("trialscope-drift")
    with mlflow.start_run(run_name=f"drift_{snap.date()}"):
        mlflow.log_metrics({f"psi.{r.feature.replace(' ', '_')}": r.psi for r in out.itertuples() if pd.notna(r.psi)})
        mlflow.log_metric("n_features_shifted", int((out.psi > 0.25).sum()))
    print(out.head(8).round(3).to_string(index=False))
    print(json.dumps({"shifted": out[out.psi > 0.25].feature.tolist()}))
    return out


if __name__ == "__main__":
    main()
