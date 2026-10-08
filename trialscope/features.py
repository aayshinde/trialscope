"""Leakage-safe feature matrix.

Every feature is registered with whether it is known at the trial's START. The current
ClinicalTrials.gov record silently overwrites planned values with actuals (enrollment becomes
ACTUAL, site lists are final, results get posted), so naive features leak the outcome.
The default model uses only start-time features; the LEAKY group exists for the ablation that
shows how badly leakage inflates metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trialscope import config, db

# name -> (known_at_start, note)
NUMERIC = {
    "n_arms": (True, "design"), "n_collaborators": (True, "can change, usually set at registration"),
    "min_age_years": (True, "eligibility"), "max_age_years": (True, "eligibility"),
    "sponsor_prior_closed": (True, "closed in the 5y before start (ASOF join)"), "sponsor_prior_terminated": (True, "5y before start"),
    "sponsor_prior_started": (True, "started in the 5y before start"), "sponsor_prior_term_rate": (True, "5y before start, smoothed"),
    "n_branches": (True, "MeSH breadth"), "n_drug_interventions": (True, "design"),
    "days_registration_to_start": (True, "registration date known at start/registration"),
    "first_planned_enrollment": (True, "from SCD2 history: first ESTIMATED value observed; mostly missing for old trials"),
    "n_inclusion": (True, "current criteria text (may be amended later — minor risk)"),
    "n_exclusion": (True, "current criteria text"), "elig_chars": (True, "current criteria text"),
    "ecog_max": (True, "criteria text"), "n_lab_thresholds": (True, "criteria text"),
}
BOOLEAN = {
    "is_rare_disease": (True, "MeSH"), "healthy_volunteers": (True, "eligibility"),
    "requires_biomarker": (True, "criteria text"), "requires_prior_therapy": (True, "criteria text"),
    "excludes_brain_mets": (True, "criteria text"),
    "drug_previously_approved": (True, "openFDA approval date < trial start (only if `enrich fda` was run)"),
}
CATEGORICAL = {
    "phase_group": (True, ""), "sponsor_class": (True, ""), "therapeutic_area": (True, ""), "allocation": (True, ""),
    "intervention_model": (True, ""), "primary_purpose": (True, ""), "masking": (True, ""), "sex": (True, ""),
}
LEAKY = {  # known only after the trial ran — used ONLY in the leakage ablation
    "enrollment_count": "becomes ACTUAL enrollment once the trial closes (terminated trials enroll few)",
    "n_locations": "final site list; sites close or never activate in failing trials",
    "n_countries": "final site list",
    "has_results": "results are posted after completion",
    "n_result_publications": "publications appear after completion",
    "n_versions": "more record updates happen over a longer life",
}
EMBED_PREFIX = "e"


def load_frame() -> pd.DataFrame:
    t = db.query("select * from marts.fct_trials where start_date is not null")
    el = db.query("select * from ml.eligibility_features")
    df = t.merge(el, on="nct_id", how="left")
    if db.table_exists("text_embeddings"):
        df = df.merge(db.query("select * from ml.text_embeddings"), on="nct_id", how="left")
    if db.table_exists("trial_drug_approval"):
        df = df.merge(db.query("select * from ml.trial_drug_approval"), on="nct_id", how="left")
    df["days_registration_to_start"] = (pd.to_datetime(df.start_date) - pd.to_datetime(df.first_submit_date)).dt.days
    if db.table_exists("termination_reasons"):
        r = db.query("select nct_id, reason from ml.termination_reasons")
        df = df.merge(r, on="nct_id", how="left")
    else:
        df["reason"] = None
    df["accrual_failure"] = np.where(df.terminated_flag.isna(), np.nan,
                                     ((df.terminated_flag == 1) & (df.reason == "enrollment")).astype(float))
    df.loc[(df.terminated_flag == 1) & (df.reason != "enrollment"), "accrual_failure"] = np.nan  # other failures excluded
    return df


class FeatureBuilder:
    """Fit category levels on the training cohort, then transform any cohort identically."""

    def __init__(self, leaky: bool = False, embeddings: bool = True, max_levels: int = 15):
        self.leaky, self.embeddings, self.max_levels = leaky, embeddings, max_levels
        self.levels: dict[str, list[str]] = {}
        self.columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        for c in CATEGORICAL:
            self.levels[c] = df[c].fillna("NA").astype(str).value_counts().head(self.max_levels).index.tolist()
        self.columns = list(self.transform(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}          # collected then concatenated once (avoids a fragmented DataFrame)
        for c in NUMERIC:
            cols[c] = pd.to_numeric(df.get(c), errors="coerce")
        for c in BOOLEAN:
            cols[c] = df[c].astype("float") if c in df else pd.Series(np.nan, index=df.index)
        for c, lv in self.levels.items():
            v = df[c].fillna("NA").astype(str)
            for level in lv:
                cols[f"{c}={level}"] = (v == level).astype(np.float32)
        if self.embeddings:
            for c in [c for c in df.columns if c.startswith(EMBED_PREFIX) and c[1:].isdigit()]:
                cols[c] = df[c]
        if self.leaky:
            for c in LEAKY:
                cols[f"LEAK:{c}"] = pd.to_numeric(df.get(c), errors="coerce").astype(float)
        X = pd.concat(cols, axis=1).astype(np.float32) if cols else pd.DataFrame(index=df.index)
        X.index = df.index
        return X.reindex(columns=self.columns, fill_value=0) if self.columns else X

    @staticmethod
    def group_of(col: str) -> str:
        """Collapse one-hot / embedding columns back to a human-readable driver name."""
        if col.startswith(EMBED_PREFIX) and col[1:].isdigit():
            return "eligibility & title text (embedding)"
        return col.split("=")[0] if "=" in col else col


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Out-of-time cohorts by start year (resolved trials only, sane durations)."""
    ok = (df.is_resolved.fillna(False) & df.terminated_flag.notna() & ~(df.duration_days < 0)
          & (df.start_year >= config.MODEL_MIN_START_YEAR))
    d = df[ok]
    return {
        "train": d[d.start_year <= config.TRAIN_END_YEAR],
        "val": d[d.start_year == config.VAL_YEAR],
        "test": d[(d.start_year >= config.TEST_START_YEAR) & (d.start_year <= config.TEST_END_YEAR)],
    }


def resolution_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Share of trials per start year that have resolved — shows why recent cohorts can't be test sets yet."""
    g = df.groupby("start_year")
    return pd.DataFrame({"n": g.size(), "resolved_share": g.is_resolved.mean().round(3),
                         "termination_rate_among_resolved": g.terminated_flag.mean().round(3)})
