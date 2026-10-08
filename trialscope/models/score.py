"""Score every trial: termination risk, accrual-failure risk, top SHAP drivers, and for active
trials P(running late) + predicted completion. Also precomputes similar historical trials.

    python -m trialscope.models.score            # writes ml.risk_scores and ml.similar_trials
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

from trialscope import config, db
from trialscope.features import FeatureBuilder, load_frame
from trialscope.models.survival import cox_design, p_late

READABLE = {
    "sponsor_prior_term_rate": "sponsor's past termination rate", "sponsor_prior_started": "sponsor's trial experience",
    "sponsor_prior_closed": "sponsor's closed trials", "sponsor_prior_terminated": "sponsor's past terminations",
    "n_exclusion": "number of exclusion criteria", "n_inclusion": "number of inclusion criteria",
    "elig_chars": "length of eligibility criteria", "requires_biomarker": "biomarker-gated eligibility",
    "requires_prior_therapy": "requires prior therapy", "excludes_brain_mets": "excludes brain metastases",
    "is_rare_disease": "rare disease", "days_registration_to_start": "registration-to-start gap",
    "first_planned_enrollment": "planned enrollment", "ecog_max": "ECOG limit", "n_lab_thresholds": "lab-value thresholds",
}


def load_pickle(name: str):
    with (config.MODEL_DIR / name).open("rb") as f:
        return pickle.load(f)


def drivers(model, X: pd.DataFrame, k: int = 3) -> list[str]:
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    groups = pd.Series([FeatureBuilder.group_of(c) for c in X.columns])
    out = []
    for row in contrib:
        s = pd.Series(row).groupby(groups.values).sum()
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(k)
        out.append(json.dumps([{"driver": READABLE.get(g, g.replace("_", " ")), "effect": "raises risk" if v > 0 else "lowers risk",
                                "shap": round(float(v), 3)} for g, v in top.items()]))
    return out


def _vec(d: pd.DataFrame, spec: dict) -> np.ndarray:
    E = d.reindex(columns=spec["emb_cols"]).fillna(0).to_numpy(np.float32)
    P = np.stack([(d.phase_group == p).to_numpy(np.float32) for p in spec["phases"]], 1) * 0.35
    A = np.stack([(d.therapeutic_area == a).to_numpy(np.float32) for a in spec["areas"]], 1) * 0.35
    return np.hstack([E, P, A])


def similar_index(df: pd.DataFrame) -> dict:
    """kNN over text embedding + phase + area, restricted to resolved trials (we know how they ended)."""
    spec = {"emb_cols": [c for c in df.columns if c.startswith("e") and c[1:].isdigit()],
            "phases": sorted(df.phase_group.dropna().unique()), "areas": sorted(df.therapeutic_area.dropna().unique())}
    hist = df[df.is_resolved.fillna(False)].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=min(40, len(hist)), metric="cosine").fit(_vec(hist, spec))
    keep = ["nct_id", "brief_title", "status_group", "why_stopped", "duration_days", "start_date"]
    return {"nn": nn, "hist": hist[keep], "spec": spec}


def find_similar(index: dict, query: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Most similar resolved trials that STARTED BEFORE the query trial (what was knowable then)."""
    hist = index["hist"]
    dist, idx = index["nn"].kneighbors(_vec(query, index["spec"]))
    rows = []
    for qi, (ds, ix) in enumerate(zip(dist, idx)):
        q = query.iloc[qi]
        cands = [(d, hist.iloc[j]) for d, j in zip(ds, ix) if hist.nct_id.iloc[j] != q.nct_id
                 and pd.Timestamp(hist.start_date.iloc[j]) < pd.Timestamp(q.start_date)]
        for rank, (d, h) in enumerate(cands[:k], 1):
            rows.append({"nct_id": q.nct_id, "rank": rank, "similar_nct_id": h.nct_id, "similarity": round(1 - float(d), 3),
                         "similar_title": h.brief_title, "similar_status": h.status_group, "similar_why_stopped": h.why_stopped,
                         "similar_duration_months": round(h.duration_days / 30.44, 1) if pd.notna(h.duration_days) else None})
    return pd.DataFrame(rows)


def similar_for(nct_id: str, k: int = 5) -> pd.DataFrame:
    """On-demand lookup for any trial (used by the dashboard and the agent)."""
    index = load_pickle("similar_index.pkl")
    q = db.query("select t.*, e.* exclude (nct_id) from marts.fct_trials t join ml.text_embeddings e using (nct_id) "
                 "where t.nct_id = ?", [nct_id.upper()])
    return find_similar(index, q, k) if len(q) else pd.DataFrame()


def main() -> None:
    df = load_frame()
    df = df[df.start_year >= 2000].reset_index(drop=True)
    term, acc, surv = load_pickle("termination.pkl"), load_pickle("accrual.pkl"), load_pickle("survival.pkl")

    X = term["features"].transform(df)
    raw = term["model"].predict_proba(X)[:, 1]
    # keep probabilities away from exactly 0 / 1
    out = pd.DataFrame({"nct_id": df.nct_id, "risk_termination": np.clip(term["calibrator"].predict(raw), 0.01, 0.99).round(4),
                        "risk_rank_score": raw.round(5)})
    out["risk_accrual"] = np.clip(acc["calibrator"].predict(acc["model"].predict_proba(acc["features"].transform(df))[:, 1]), 0.01, 0.99).round(4)
    out["drivers"] = drivers(term["model"], X)

    active = (df.status_group == "active").to_numpy()
    out["risk_percentile_active"] = np.nan
    out.loc[active, "risk_percentile_active"] = pd.Series(raw[active]).rank(pct=True).round(3).to_numpy()

    # delay: only for active trials with a planned completion date
    has_plan = active & df.planned_primary_completion.notna().to_numpy() & df.start_date.notna().to_numpy()
    out["p_late"], out["predicted_completion"], out["expected_slip_days"] = np.nan, pd.NaT, np.nan
    if has_plan.any():
        d = df[has_plan].copy()
        snap = pd.Timestamp(d.snapshot_date.iloc[0])
        elapsed = ((snap - pd.to_datetime(d.start_date)).dt.days / 30.44).to_numpy()
        planned = ((pd.to_datetime(d.planned_primary_completion) - pd.to_datetime(d.start_date)).dt.days / 30.44).to_numpy()
        Xc, _ = cox_design(d.assign(n_exclusion=d.n_exclusion), surv["levels"])
        Xc = Xc.reindex(columns=surv["cox"].params_.index, fill_value=0)
        p, med = p_late(surv["cox"], Xc, elapsed, planned)
        p = np.where(planned < elapsed, 1.0, p)            # already past the planned date
        pred = pd.to_datetime(d.start_date) + pd.to_timedelta(med * 30.44, unit="D")
        out.loc[has_plan, "p_late"] = p.round(3)
        out.loc[has_plan, "predicted_completion"] = pred.dt.floor("D").to_numpy()
        out.loc[has_plan, "expected_slip_days"] = (pred - pd.to_datetime(d.planned_primary_completion)).dt.days.to_numpy()
    out["scored_at"] = datetime.now()
    db.write_table(out, "risk_scores")

    index = similar_index(df)
    sim = find_similar(index, df[active])
    db.write_table(sim if len(sim) else pd.DataFrame(columns=["nct_id"]), "similar_trials")
    with (config.MODEL_DIR / "similar_index.pkl").open("wb") as f:
        pickle.dump(index, f)
    print(f"scored {len(out):,} trials ({active.sum():,} active, {has_plan.sum():,} with a planned completion date); "
          f"similar trials for {sim.nct_id.nunique() if len(sim) else 0:,} active trials")


if __name__ == "__main__":
    main()
