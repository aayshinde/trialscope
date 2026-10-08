"""Offline tests: API client (mocked), parsing, entity resolution, dbt + SCD2, leakage guards,
text features, models, scoring, agent guards and evaluation logic.  Run:  pytest -q
"""
from __future__ import annotations

import gzip
import io
import json

import numpy as np
import pandas as pd
import pytest

from trialscope import config

REAL_SHAPE = {  # mirrors the structure of a real v2 record (NCT02008227)
    "protocolSection": {
        "identificationModule": {"nctId": "NCT02008227", "briefTitle": "A Study of Atezolizumab vs Docetaxel in NSCLC"},
        "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2014-03-11", "type": "ACTUAL"},
                         "primaryCompletionDateStruct": {"date": "2016-07", "type": "ACTUAL"},
                         "completionDateStruct": {"date": "2019-01-09", "type": "ACTUAL"}, "studyFirstSubmitDate": "2013-12-06",
                         "lastUpdatePostDateStruct": {"date": "2019-12-20", "type": "ACTUAL"}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Hoffmann-La Roche", "class": "INDUSTRY"}},
        "conditionsModule": {"conditions": ["Non-Squamous Non-Small Cell Lung Cancer"]},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"],
                         "designInfo": {"allocation": "RANDOMIZED", "interventionModel": "PARALLEL", "primaryPurpose": "TREATMENT",
                                        "maskingInfo": {"masking": "NONE"}}, "enrollmentInfo": {"count": 1225, "type": "ACTUAL"}},
        "armsInterventionsModule": {"armGroups": [{"label": "Docetaxel"}, {"label": "Atezolizumab"}],
                                    "interventions": [{"type": "DRUG", "name": "Atezolizumab"}, {"type": "DRUG", "name": "Docetaxel"}]},
        "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n\n* Locally advanced or metastatic NSCLC\n\n"
                                                     "Exclusion Criteria:\n\n* Active brain metastases", "healthyVolunteers": False,
                              "minimumAge": "18 Years"},
        "contactsLocationsModule": {"locations": [{"facility": "Kaiser Permanente - Hayward", "city": "Hayward", "country": "United States"},
                                                  {"facility": "X", "city": "Y", "country": "Germany"}]},
        "referencesModule": {"references": [{"pmid": "27979383", "type": "DERIVED"}]},
    },
    "derivedSection": {"conditionBrowseModule": {"meshes": [{"id": "D002289", "term": "Carcinoma, Non-Small-Cell Lung"}],
                                                 "browseBranches": [{"abbrev": "BC04", "name": "Neoplasms"}]}},
    "hasResults": True,
}


# ------------------------------------------------------------------ parsing & client
def test_parse_real_shape():
    from trialscope.ingest.parse import parse_age, parse_date, parse_study

    p = parse_study(REAL_SHAPE)
    s = p["studies"][0]
    assert s["nct_id"] == "NCT02008227" and s["phase"] == "PHASE3" and s["lead_sponsor_class"] == "INDUSTRY"
    assert str(s["primary_completion_date"]) == "2016-07-01" and s["min_age_years"] == 18 and s["max_age_years"] is None
    assert s["n_locations"] == 2 and s["n_countries"] == 2 and s["has_results"] is True
    assert len(p["interventions"]) == 2 and p["browse_branches"][0]["name"] == "Neoplasms"
    assert parse_date("2019") is not None and parse_date("bad") is None and parse_age("6 Months") == 0.5


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_ctgov_pull_paginates_and_falls_back(monkeypatch):
    from trialscope.ingest import ctgov

    calls = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls.append(dict(params))
        if "fields" in params:
            return FakeResp(400, {"error": "bad field"})
        if "pageToken" not in params:
            return FakeResp(200, {"studies": [REAL_SHAPE], "nextPageToken": "abc", "totalCount": 2})
        return FakeResp(200, {"studies": [REAL_SHAPE]})

    monkeypatch.setattr("requests.Session.get", fake_get)
    out = ctgov.pull("full", resume=False)
    assert out["n"] == 2
    assert "AREA[StudyType]INTERVENTIONAL" in calls[0]["filter.advanced"]
    assert "fields" not in calls[-1] and calls[-1]["pageToken"] == "abc"
    assert not json.loads(ctgov.STATE.read_text())["in_progress"]
    assert "LastUpdatePostDate" in ctgov.base_filter("2026-10-01")
    (config.LANDING_DIR / out["file"].split("/")[-1].split("\\")[-1]).unlink()


# ------------------------------------------------------------------ entity resolution
def test_sponsor_normalization_and_gold():
    from trialscope.entity import evaluate, normalize, resolve

    assert normalize("Pfizer, Inc.") == normalize("PFIZER INC") == "pfizer"
    assert normalize("M.D. Anderson Cancer Center") == normalize("MD Anderson Cancer Center")
    assert normalize("University of California, San Francisco") != normalize("University of California, Los Angeles")
    m = resolve(pd.Series(["Merck Sharp & Dohme LLC", "Merck Sharp and Dohme", "Merck KGaA"]), use_aliases=False)
    assert m.set_index("raw_name").sponsor_key["Merck KGaA"] != m.set_index("raw_name").sponsor_key["Merck Sharp and Dohme"]
    res = evaluate()
    assert res["rules_only"]["pairs_precision"] == 1.0          # rules never merge different sponsors on the gold set


# ------------------------------------------------------------------ full demo pipeline
@pytest.fixture(scope="session")
def pipeline():
    import sys

    from trialscope import demo_data, entity, insights
    from trialscope.dbt_runner import dbt
    from trialscope.ingest.load import load_all
    from trialscope.models import classifier, drift, score, survival
    from trialscope.text import eligibility, termination

    studies = demo_data.generate(4000)
    demo_data.write(studies, "demo_full_20261001")
    assert load_all() == 4000
    entity.build()
    assert dbt("build")
    demo_data.write(demo_data.make_update(studies, np.random.default_rng(1)), "demo_incr_20261008")
    load_all()
    assert dbt("build")
    eligibility.build()
    termination.build()
    sys.argv = ["x"]
    classifier.main()
    survival.main()
    score.main()
    drift.main()
    insights.main()
    return True


def test_load_is_idempotent(pipeline):
    from trialscope.ingest.load import load_all

    assert load_all() == 0


def test_scd2_history_and_dbt_tests(pipeline):
    from trialscope import db

    v = db.query("select count(*) n, count(distinct nct_id) t from snapshots.snap_studies").iloc[0]
    assert v.n > v.t                       # the update batch opened new versions
    h = db.query("select sum(n_completion_slips) s, sum(n_status_changes) c from marts.fct_trials").iloc[0]
    assert h.s > 0 and h.c > 0
    new = db.query("select min(dbt_valid_from) d from snapshots.snap_studies where dbt_valid_to is null and nct_id in "
                   "(select nct_id from intermediate.int_study_history where n_versions > 1)").d[0]
    assert str(new)[:10] == "2026-10-08"   # valid_from = registry update date, not load time


def test_no_leaky_features_in_model(pipeline):
    import pickle

    from trialscope.features import LEAKY

    fb = pickle.load((config.MODEL_DIR / "termination.pkl").open("rb"))["features"]
    assert not any(c.startswith("LEAK:") or c in LEAKY for c in fb.columns)
    m = json.loads((config.RESULTS_DIR / "classifier_metrics.json").read_text())
    assert m["termination"]["xgboost_WITH_LEAKY_FEATURES"]["auroc"] > m["termination"]["xgboost"]["auroc"]


def test_out_of_time_split(pipeline):
    from trialscope.features import load_frame, split

    d = split(load_frame())
    assert d["train"].start_year.max() <= config.TRAIN_END_YEAR < d["val"].start_year.min() <= d["test"].start_year.min()
    assert d["train"].start_year.min() >= config.MODEL_MIN_START_YEAR


def test_scores_and_similar(pipeline):
    from trialscope import db
    from trialscope.models.score import similar_for

    r = db.query("select r.*, t.status_group, t.start_date from ml.risk_scores r join marts.fct_trials t using (nct_id)")
    assert r.risk_termination.between(0.01, 0.99).all()
    assert r[r.status_group == "active"].p_late.notna().all()
    nct = r[r.status_group == "active"].nct_id.iloc[0]
    sim = similar_for(nct, 5)
    start = r.set_index("nct_id").start_date
    assert len(sim) == 5 and all(pd.Timestamp(start[s]) < pd.Timestamp(start[nct]) for s in sim.similar_nct_id)
    assert json.loads(r.drivers.iloc[0])[0]["effect"] in ("raises risk", "lowers risk")
    for f in ("classifier_report.md", "survival_report.md", "drift.md", "insights.md", "termination_reasons.md"):
        assert (config.RESULTS_DIR / f).exists()


# ------------------------------------------------------------------ text
def test_eligibility_rules():
    from trialscope.text.eligibility import rule_features

    f = rule_features("Inclusion Criteria:\n\n* Documented EGFR mutation\n* At least one prior line of platinum chemotherapy\n"
                      "* ECOG performance status 0-2\n\nExclusion Criteria:\n\n1. Untreated brain metastases\n2. Pregnancy")
    assert (f["n_inclusion"], f["n_exclusion"], f["ecog_max"]) == (3, 2, 2)
    assert f["requires_biomarker"] and f["requires_prior_therapy"] and f["excludes_brain_mets"]


def test_termination_rules():
    from trialscope.text.termination import classify_rule

    assert classify_rule("Slow accrual") == "enrollment"
    assert classify_rule("Low enrollment due to COVID-19 pandemic") == "covid"
    assert classify_rule("Company decision not related to safety") == "business"     # negation handled
    assert classify_rule("Business decision; no safety concerns") == "business"
    assert classify_rule("Unacceptable toxicity") == "safety"
    assert classify_rule("Business decision unrelated to efficacy or safety") == "business"
    assert classify_rule("Futility at interim analysis") == "efficacy"
    assert classify_rule("PI left the institution") == "logistics"
    assert classify_rule("Lack of funding") == "funding"


def test_llm_extraction_eval(pipeline, monkeypatch):
    from trialscope import db
    from trialscope.text import llm_extract

    t = db.query("select nct_id, eligibility_criteria from staging.stg_studies limit 5")
    from trialscope.text.eligibility import rule_features
    labels = [{"nct_id": n, **{k: rule_features(x)[k] for k in llm_extract.FIELDS}} for n, x in zip(t.nct_id, t.eligibility_criteria)]
    p = config.LABELS_DIR / "eligibility_labels.jsonl"
    backup = p.read_text() if p.exists() else None
    p.write_text("\n".join(json.dumps(l) for l in labels))
    try:
        fake = pd.DataFrame(labels).assign(n_exclusion=lambda d: d.n_exclusion + 1, biomarkers="[]")
        db.write_table(fake, "llm_eligibility")
        res = llm_extract.evaluate()
        assert res["fields"]["n_inclusion"]["rules_acc"] == 1.0 and res["fields"]["n_exclusion"]["llm_acc"] == 0.0
        assert res["fields"]["n_exclusion"]["llm_within1"] == 1.0
    finally:
        p.unlink() if backup is None else p.write_text(backup)


# ------------------------------------------------------------------ agent
def test_sql_guard(pipeline):
    from trialscope.agent import check_sql, run_sql

    for bad in ["delete from marts.fct_trials", "select 1; drop table x", "select * from raw.studies",
                "select * from read_csv('x.csv')", "attach 'x.db'", "select * from marts.fct_trials; select 2"]:
        with pytest.raises(ValueError):
            check_sql(bad)
    out = run_sql("select status_group, count(*) n from marts.fct_trials group by 1 order by 2 desc")
    assert out.startswith("status_group,n")
    assert run_sql("update marts.fct_trials set x=1").startswith("ERROR")


def test_agent_loop_and_citation_check(pipeline):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda

    from trialscope.agent import ask

    class FakeLLM:
        def bind_tools(self, tools):
            def respond(msgs):
                if not any(isinstance(m, ToolMessage) for m in msgs):
                    return AIMessage("", tool_calls=[{"name": "risk_leaderboard", "id": "1",
                                                      "args": {"phase_group": "Phase 2", "therapeutic_area": "Neoplasms", "n": 3}}])
                ids = [l.split(",")[0] for l in msgs[-1].content.splitlines()[1:3]]
                return AIMessage(f"Highest risk: {ids[0]} and {ids[1]}; also NCT99999999.")
            return RunnableLambda(respond)

    r = ask("Which Phase 2 oncology trials are most likely to stall?", llm=FakeLLM())
    assert r["tool_calls"][0]["name"] == "risk_leaderboard"
    assert len(r["cited_trials"]) == 3 and r["uncited_ids"] == ["NCT99999999"]   # invented ID is caught


def test_agent_eval_scoring():
    from trialscope.agent_eval import correct

    assert correct({"family": "count", "gold": 12}, "There were 12 terminated trials.")
    assert correct({"family": "rate", "gold": 23.4}, "About 23.0% were terminated")
    assert correct({"family": "lookup", "gold": 0.31}, "risk is 31%")
    assert not correct({"family": "top_risk", "gold": "NCT01234567"}, "NCT07654321 then NCT01234567")


# ------------------------------------------------------------------ enrichment parsers
def test_pubmed_and_fda_parsers():
    from trialscope.ingest.enrich import first_approval, parse_efetch

    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>27979383</PMID><Article><Journal><JournalIssue>
             <PubDate><Year>2017</Year></PubDate></JournalIssue></Journal><DataBankList><DataBank>
             <DataBankName>ClinicalTrials.gov</DataBankName><AccessionNumberList><AccessionNumber>NCT02008227</AccessionNumber>
             <AccessionNumber>NCT00000001</AccessionNumber></AccessionNumberList></DataBank></DataBankList></Article>
             </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    assert parse_efetch(xml, {"NCT02008227"}) == [{"nct_id": "NCT02008227", "pmid": "27979383", "pub_year": 2017}]
    res = [{"application_number": "BLA125514", "submissions": [
        {"submission_type": "SUPPL", "submission_status": "AP", "submission_status_date": "20150101"},
        {"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": "20140904"}]}]
    assert first_approval(res) == ("20140904", "BLA125514")


# ------------------------------------------------------------------ added with the portfolio polish
def test_branches_fallback_chain():
    from trialscope.ingest.parse import branches_from_mesh

    names = lambda cb, cond=None: [b["name"] for b in branches_from_mesh(cb, cond)]
    assert names({"browseBranches": [{"abbrev": "BC04", "name": "Neoplasms"}]}) == ["Neoplasms"]          # API value wins
    mesh = {"meshes": [{"term": "Influenza, Human"}], "ancestors": [{"term": "Virus Diseases"}, {"term": "Respiratory Tract Diseases"}]}
    assert names(mesh) == ["Infections", "Respiratory Tract (Lung and Bronchial) Diseases"]               # MeSH ancestors
    assert names({}, ["Metastatic Breast Cancer"]) == ["Neoplasms"]                                      # keyword fallback
    assert names({}, ["Healthy Volunteers"]) == ["Healthy Volunteers"]
    assert names({}, ["zzz"]) == []
    assert "Rare Diseases" in names({"ancestors": [{"term": "Genetic Diseases, Inborn"}]})               # documented proxy


def test_cox_design_drops_constant_rare_flag():
    from trialscope.models.survival import cox_design

    d = pd.DataFrame({"phase_group": ["Phase 1", "Phase 2"] * 5, "sponsor_class": ["NIH", "OTHER"] * 5,
                      "therapeutic_area": ["Neoplasms", "Infections"] * 5, "is_rare_disease": [False] * 10, "n_exclusion": 3,
                      "requires_biomarker": False, "sponsor_prior_term_rate": 0.1, "sponsor_prior_started": 4})
    X, levels = cox_design(d)
    assert "is_rare_disease" not in X.columns and levels["_use_rare"] is False        # constant column would make Cox singular
    X2, _ = cox_design(d.assign(is_rare_disease=[True, False] * 5))
    assert "is_rare_disease" in X2.columns


def test_platt_calibrator_is_smooth_and_bounded():
    from trialscope.models.calibration import PlattCalibrator

    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 0.6, 1000)                                                  # XGBoost scores here top out near 0.6
    y = (rng.uniform(0, 1, 1000) < raw * 0.5).astype(int)
    cal = PlattCalibrator().fit(raw, y)
    grid = cal.predict(np.linspace(0, 0.6, 50))
    assert np.all(np.diff(grid) >= -1e-9)                                            # monotone
    assert 0 < grid.min() and grid.max() < 0.6                                       # no 0/1 plateau inside the observed range
    assert 0 < cal.predict([5.0])[0] <= 1 and 0 <= cal.predict([-5.0])[0] < 1        # out-of-range scores stay valid probabilities


def test_df_text_keeps_drivers_json_intact(pipeline):
    from trialscope.agent import _df_text

    drivers = json.dumps([{"driver": "x" * 90, "effect": "raises risk", "shap": 0.5}] * 3)
    out = _df_text(pd.DataFrame({"nct_id": ["NCT1"], "title": ["t" * 400], "drivers": [drivers]}))
    back = pd.read_csv(io.StringIO(out))
    assert json.loads(back.drivers[0]) == json.loads(drivers)                         # JSON survives the round trip intact
    assert len(back.title[0]) == 160                                                  # plain text columns are still truncated


def test_llm_configured(monkeypatch):
    from trialscope.llm import llm_configured

    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert not llm_configured()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-...")                                   # the .env.example placeholder
    assert not llm_configured()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-" + "a" * 40)
    assert llm_configured()
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")               # free local model
    assert llm_configured()


def test_ask_offline_answers_without_an_llm(pipeline):
    from trialscope.agent import ask_offline

    risk = ask_offline("Which Phase 2 oncology trials starting this year are most likely to stall, and why?")
    assert risk is not None and {"nct_id", "risk_termination", "top_driver"} <= set(risk["table"].columns)
    areas = ask_offline("Which therapeutic areas had the highest termination rate for trials started 2010-2015?")
    assert areas is not None and areas["table"].termination_rate.is_monotonic_decreasing
    reasons = ask_offline("What are the most common reasons Phase 3 trials were terminated?")
    assert reasons is not None and abs(reasons["table"].share.sum() - 1) < 0.01
    assert ask_offline("tell me a joke") is None


def test_text_embedder_reproduces_pipeline_embeddings(pipeline):
    from trialscope import db
    from trialscope.text.eligibility import embed_text

    r = db.query("select s.nct_id, s.brief_title, s.eligibility_criteria, t.therapeutic_area from staging.stg_studies s "
                 "join marts.fct_trials t using (nct_id) limit 3")
    for row in r.itertuples():
        stored = db.query("select * from ml.text_embeddings where nct_id = ?", [row.nct_id]).drop(columns="nct_id").to_numpy()[0]
        assert np.allclose(stored, embed_text(row.brief_title, row.therapeutic_area, row.eligibility_criteria), atol=1e-6)


def test_trim_exclusions_keeps_inclusions():
    from trialscope.text.eligibility import rule_features, trim_exclusions

    text = "Inclusion Criteria:\n* Age 18+\n* Confirmed diagnosis\nExclusion Criteria:\n* Pregnancy\n* Prior therapy\n* Active infection\n* Other"
    out = trim_exclusions(text, 2)
    f = rule_features(out)
    assert f["n_inclusion"] == 2 and f["n_exclusion"] == 2
    assert trim_exclusions("no sections here", 2) == "no sections here"


def test_calibration_table_and_conformal_coverage():
    from trialscope.models.classifier import calibration_table, conformal_summary

    rng = np.random.default_rng(1)
    p = rng.beta(1, 8, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)                                    # perfectly calibrated by construction
    cal = calibration_table(y, p)
    assert len(cal) == 8 and (cal.lo <= cal.observed).all() and (cal.observed <= cal.hi).all()
    assert abs((cal.predicted - cal.observed).abs().mean()) < 0.03
    cf = conformal_summary(y[:2000], p[:2000], y[2000:], p[2000:], alpha=0.1)
    assert cf["coverage"] >= 0.85                                                    # target 90%, finite-sample slack


def test_watchlist(pipeline):
    from trialscope import watchlist

    df = watchlist.build(5)
    assert len(df) <= 5 and df.risk_termination.is_monotonic_decreasing or len(df) <= 1
    watchlist.main()
    watchlist.main()                                                                 # second run diffs against the first
    md = (config.RESULTS_DIR / "watchlist.md").read_text()
    assert md.startswith("# TrialScope watchlist") and "clinicaltrials.gov/study/" in md


def test_dashboard_smoke(pipeline):
    """Every tab renders against the pipeline output, the offline Ask answer works, and the what-if tool re-scores edited text."""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Overview", "Trial lookup", "What-if", "Portfolio risk", "Ask TrialScope", "Insights",
                                          "Model & data quality"]
    labels = [m.label for m in at.metric]
    assert "Out-of-time AUROC" in labels and "Termination risk" in labels
    [b for b in at.button if b.label == "Ask"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    [b for b in at.button if b.label.startswith("Keep only the first 5")][0].click().run()   # what-if: edit text, re-score
    assert not at.exception, [e.value for e in at.exception]
