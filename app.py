"""TrialScope dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from trialscope import config, db
from trialscope.text.eligibility import trim_exclusions

st.set_page_config(page_title="TrialScope", page_icon="🧪", layout="wide")

# A hosted copy has no pipeline run: fall back to the slim bundle committed under deploy/.
DEPLOY = Path(__file__).resolve().parent / "deploy"
if not config.WAREHOUSE.exists() and (DEPLOY / "trialscope_demo.duckdb").exists():
    config.WAREHOUSE = DEPLOY / "trialscope_demo.duckdb"
    config.RESULTS_DIR = DEPLOY / "results"
    config.MODEL_DIR = DEPLOY / "models"

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.hero h1 {font-size: 2.6rem; margin-bottom: 0.1rem; letter-spacing: -0.02em;}
.hero p {font-size: 1.1rem; color: #5b6675; margin-top: 0;}
div[data-testid="stMetric"] {background: #f4f6fb; border: 1px solid #e3e8f2; padding: 14px 16px; border-radius: 12px;}
div[data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #5b6675;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; color:#fff;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def con():
    return db.connect(read_only=True)


@st.cache_data(ttl=600)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return con().execute(sql, list(params)).df()


def pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.0%}"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def risk_pill(p: float, base: float) -> str:
    """Plain-language risk band relative to the historical base rate."""
    if pd.isna(p):
        return ""
    ratio = p / base if base else 1
    label, colour = (("HIGH", "#d64545") if ratio >= 2 else ("ELEVATED", "#e08a1e") if ratio >= 1.25
                     else ("TYPICAL", "#5b6675") if ratio >= 0.6 else ("LOW", "#2e9e6b"))
    return f'<span class="pill" style="background:{colour}">{label} · {ratio:.1f}× base rate</span>'


if not config.WAREHOUSE.exists():
    st.error("No warehouse yet. Run `python run_pipeline.py --demo` (or the real pipeline) first.")
    st.stop()

meta = q("select max(snapshot_date) snap, count(*) n, count(*) filter (where status_group='active') n_active from marts.fct_trials").iloc[0]
metrics = read_json(config.RESULTS_DIR / "classifier_metrics.json")
base_rate = float(q("select avg(terminated_flag) r from marts.fct_trials where terminated_flag is not null "
                    "and start_year between ? and ?", (config.MODEL_MIN_START_YEAR, config.TEST_END_YEAR)).r[0])

st.markdown('<div class="hero"><h1>🧪 TrialScope</h1>'
            '<p>Predicts which clinical trials will be terminated, stall on recruitment or run late, explains why, and '
            'answers questions about the portfolio. Built on public ClinicalTrials.gov data.</p></div>', unsafe_allow_html=True)
st.caption(f"{meta.n:,} interventional trials · {meta.n_active:,} active · data as of {str(meta.snap)[:10]}")

tab_over, tab_trial, tab_what, tab_port, tab_ask, tab_insight, tab_model = st.tabs(
    ["Overview", "Trial lookup", "What-if", "Portfolio risk", "Ask TrialScope", "Insights", "Model & data quality"])

# ---------------------------------------------------------------- overview
with tab_over:
    tm = metrics.get("termination", {})
    xgb_m, base_m = tm.get("xgboost", {}), tm.get("phase_x_area_rate", {})
    k = st.columns(4)
    k[0].metric("Trials analysed", f"{meta.n:,}", help="Interventional trials started since 2000 in the warehouse.")
    k[1].metric("Historical termination rate", pct(base_rate), help="Resolved trials started 2008-2019 that ended terminated.")
    if xgb_m:
        k[2].metric("Out-of-time AUROC", f"{xgb_m['auroc']:.2f}",
                    delta=f"{xgb_m['auroc'] - base_m.get('auroc', 0):+.2f} vs phase×area baseline" if base_m else None,
                    help="Trained on trials started up to 2015, tested on 2017-2019 trials the model never saw.")
        k[3].metric("Terminated trials caught in top 10%", pct(xgb_m["flagged_top10"]),
                    delta=f"{xgb_m['flagged_top10'] / 0.10:.1f}× random", help="Share of trials that eventually terminated "
                    "that were in the model's highest-risk decile at launch.")
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("Where trials fail")
        area = q("""select therapeutic_area, count(*) n, avg(terminated_flag) rate from marts.fct_trials
                    where terminated_flag is not null and start_year between ? and ? and therapeutic_area <> 'Other / unspecified'
                    group by 1 having count(*) >= 100 order by 3 desc limit 12""", (config.MODEL_MIN_START_YEAR, config.TEST_END_YEAR))
        chart = alt.Chart(area).mark_bar(color="#2f6fed", cornerRadiusEnd=4).encode(
            x=alt.X("rate:Q", axis=alt.Axis(format="%", title="termination rate")),
            y=alt.Y("therapeutic_area:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["therapeutic_area", alt.Tooltip("rate:Q", format=".1%"), alt.Tooltip("n:Q", title="resolved trials")]
        ).properties(height=360)
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Highest-risk active trials")
        top = q("""select t.nct_id, t.brief_title as title, t.phase_group as phase, t.therapeutic_area as area,
                          r.risk_termination as termination_risk, r.p_late as p_late
                   from marts.fct_trials t join ml.risk_scores r using (nct_id)
                   where t.status_group = 'active' and t.start_year >= year(t.snapshot_date) - 2
                   order by r.risk_rank_score desc limit 10""")
        st.dataframe(top, hide_index=True, width="stretch", height=380,
                     column_config={"termination_risk": st.column_config.ProgressColumn("termination risk", min_value=0, max_value=1, format="%.0f%%"),
                                    "p_late": st.column_config.ProgressColumn("P(late)", min_value=0, max_value=1, format="%.2f")})
        st.caption("Open any NCT ID in the **Trial lookup** tab to see why it scores high.")
    with st.expander("How to read this dashboard"):
        st.markdown(
            f"- **Termination risk** is a calibrated probability that a trial ends terminated instead of completed. "
            f"The historical base rate is **{base_rate:.0%}**, so 30% means roughly {0.30 / base_rate:.0f}× the usual risk.\n"
            "- Only information known **when the trial starts** is used. Fields that change as a trial runs (actual enrollment, final site "
            "count, posted results) are excluded because they leak the answer. The *Model & data quality* tab shows how much "
            "that leakage would inflate the score.\n"
            "- Scores are model estimates, not facts. See the calibration chart for how much to trust them.")

# ---------------------------------------------------------------- trial lookup
with tab_trial:
    default = q("select nct_id from ml.risk_scores r join marts.fct_trials t using (nct_id) where t.status_group='active' "
                "order by risk_rank_score desc limit 1").nct_id.iloc[0]
    nct = st.text_input("NCT ID", value=default).strip().upper()
    t = q("""select t.*, r.risk_termination, r.risk_accrual, r.risk_percentile_active, r.p_late, r.predicted_completion,
                    r.expected_slip_days, r.drivers
             from marts.fct_trials t left join ml.risk_scores r using (nct_id) where t.nct_id = ?""", (nct,))
    if t.empty:
        st.warning("Trial not found (only interventional trials in the warehouse).")
    else:
        r = t.iloc[0]
        st.subheader(r.brief_title)
        st.write(f"**{r.phase_group}** · {r.therapeutic_area}{' · inherited/rare disease' if r.is_rare_disease else ''} · "
                 f"sponsor **{r.sponsor_name}** ({r.sponsor_class}) · status **{r.overall_status}** · started {r.start_date}  "
                 f"· [open on ClinicalTrials.gov](https://clinicaltrials.gov/study/{r.nct_id})")
        st.markdown(risk_pill(r.risk_termination, base_rate), unsafe_allow_html=True)
        c = st.columns(4)
        c[0].metric("Termination risk", pct(r.risk_termination),
                    help="Calibrated probability that the trial ends terminated rather than completed (out-of-time validated model).")
        c[1].metric("Accrual-failure risk", pct(r.risk_accrual), help="Probability of termination due to low enrollment.")
        c[2].metric("P(misses planned completion)", pct(r.p_late))
        c[3].metric("Expected slip", "–" if pd.isna(r.expected_slip_days) else f"{r.expected_slip_days / 30.44:+.0f} months",
                    help="Cox-model median completion minus the currently planned primary completion date.")
        st.caption(f"For context, {base_rate:.0%} of resolved trials ended terminated.")
        if r.status_group != "active":
            st.info(f"This trial is already {r.status_group}. Scores show what the model would have said at its start."
                    + (f" Reason given: *{r.why_stopped}*" if r.why_stopped else ""))
        left, right = st.columns(2)
        with left:
            st.markdown("**Top risk drivers** (SHAP)")
            for d in json.loads(r.drivers or "[]"):
                st.markdown(f"- {'🔺' if d['effect'] == 'raises risk' else '🔻'} {d['driver']} — {d['effect']} ({d['shap']:+.2f})")
            st.markdown("**History captured by the pipeline (SCD2)**")
            hist = q("select dbt_valid_from::date as from_date, overall_status, primary_completion_date, primary_completion_type, "
                     "enrollment_count, enrollment_type from snapshots.snap_studies where nct_id = ? order by dbt_valid_from", (nct,))
            st.dataframe(hist, hide_index=True, width="stretch")
        with right:
            st.markdown("**Most similar earlier trials and how they ended**")
            sim = pd.DataFrame()
            try:
                from trialscope.models.score import similar_for
                sim = similar_for(nct, 6)
            except Exception:       # a hosted copy has no similarity index: use the precomputed table for active trials
                sim = q("select * from ml.similar_trials where nct_id = ? order by rank limit 6", (nct,))
            if len(sim):
                st.dataframe(sim[["similar_nct_id", "similarity", "similar_status", "similar_why_stopped",
                                  "similar_duration_months", "similar_title"]], hide_index=True, width="stretch")
                st.caption(f"{(sim.similar_status == 'terminated').mean():.0%} of these were terminated.")
            else:
                st.caption("No similar trials available for this one.")

# ---------------------------------------------------------------- what-if
with tab_what:
    st.write("**Protocol risk check.** Edit a trial's title and eligibility criteria and see how the model's estimate moves. "
             "The text is re-parsed and re-embedded exactly as in the pipeline. This is a sensitivity tool for the model, not causal advice.")
    term = None
    try:
        from trialscope.features import load_frame
        from trialscope.models.score import drivers as shap_drivers, load_pickle
        from trialscope.text.eligibility import embed_text, rule_features

        @st.cache_data(show_spinner="Loading feature matrix…")
        def feature_frame() -> pd.DataFrame:
            return load_frame().set_index("nct_id", drop=False)

        @st.cache_resource
        def bundles():
            return load_pickle("termination.pkl"), load_pickle("accrual.pkl")

        term, acc = bundles()
        ff = feature_frame()
        embed_text("t", "a", "c")          # fail early if the embedder is missing
    except Exception as e:
        term = None
        st.info(f"The what-if tool needs the trained models and text embedder, which are not available here ({type(e).__name__}).")
    if term is not None:
        base_id = st.text_input("Start from trial", value=default, key="wi_nct").strip().upper()
        src = q("select brief_title, eligibility_criteria from staging.stg_studies where nct_id = ?", (base_id,)) if base_id in ff.index else pd.DataFrame()
        if src.empty:
            st.warning("Trial not found." if base_id not in ff.index else
                       "The eligibility text for this trial is not included in this copy. Start from one of the highest-risk active trials.")
        else:
            row = ff.loc[[base_id]].copy()
            b = row.iloc[0]
            if st.session_state.get("wi_base") != base_id:           # new starting trial: reset the editable fields
                st.session_state.update(wi_base=base_id, wi_title=src.brief_title.iloc[0] or "",
                                        wi_crit=src.eligibility_criteria.iloc[0] or "",
                                        wi_prior=float(np.clip(b.sponsor_prior_term_rate if pd.notna(b.sponsor_prior_term_rate) else 0.15, 0, 0.6)))
            lv = term["features"].levels

            def opts(col: str, cur) -> list:
                o = [x for x in lv[col] if x != "NA"]
                return o if cur in o else [cur] + o

            def pick(label: str, col: str, cur) -> str:
                o = opts(col, cur)
                return st.selectbox(label, o, index=o.index(cur))

            left, right = st.columns([1.35, 1])
            with left:
                st.text_input("Title", key="wi_title")
                st.text_area("Eligibility criteria", key="wi_crit", height=300)
                st.button("Keep only the first 5 exclusion criteria", on_click=lambda: st.session_state.update(
                    wi_crit=trim_exclusions(st.session_state.wi_crit, 5)))
            with right:
                phase = pick("Phase", "phase_group", b.phase_group)
                area_sel = pick("Therapeutic area", "therapeutic_area", b.therapeutic_area)
                sponsor_class = pick("Sponsor type", "sponsor_class", b.sponsor_class)
                prior = st.slider("Sponsor's past termination rate", 0.0, 0.6, step=0.01, key="wi_prior")
                healthy = st.checkbox("Enrolls healthy volunteers", bool(b.healthy_volunteers) if pd.notna(b.healthy_volunteers) else False)

            new = row.copy()
            new["phase_group"], new["sponsor_class"], new["therapeutic_area"] = phase, sponsor_class, area_sel
            new["sponsor_prior_term_rate"], new["healthy_volunteers"] = prior, float(healthy)
            feats = rule_features(st.session_state.wi_crit)
            for k, v in feats.items():
                new[k] = np.nan if v is None else float(v)
            emb = embed_text(st.session_state.wi_title, area_sel, st.session_state.wi_crit)
            for i, v in enumerate(emb):
                new[f"e{i}"] = float(v)

            def score(frame: pd.DataFrame):
                X = term["features"].transform(frame)
                raw = term["model"].predict_proba(X)[:, 1]
                p = float(np.clip(term["calibrator"].predict(raw)[0], 0.01, 0.99))
                pa = float(np.clip(acc["calibrator"].predict(acc["model"].predict_proba(acc["features"].transform(frame))[:, 1])[0], 0.01, 0.99))
                return p, pa, json.loads(shap_drivers(term["model"], X)[0])

            p0, a0, _ = score(row)
            p1, a1, d1 = score(new)
            st.divider()
            m = st.columns(3)
            def delta(new_v: float, old_v: float) -> dict:
                if abs(new_v - old_v) < 0.0005:
                    return {"delta": f"unchanged from original ({pct(old_v)})", "delta_color": "off", "delta_arrow": "off"}
                return {"delta": f"{(new_v - old_v) * 100:+.1f} pts vs original ({pct(old_v)})", "delta_color": "inverse"}

            m[0].metric("Termination risk", pct(p1), **delta(p1, p0))
            m[1].metric("Accrual-failure risk", pct(a1), **delta(a1, a0))
            m[2].markdown("**Main drivers of the new estimate**")
            for d in d1:
                m[2].markdown(f"- {'🔺' if d['effect'] == 'raises risk' else '🔻'} {d['driver']} ({d['shap']:+.2f})")
            st.caption(f"Parsed from your text: {feats['n_inclusion']} inclusion and {feats['n_exclusion']} exclusion criteria"
                       f"{', biomarker-gated' if feats['requires_biomarker'] else ''}"
                       f"{', requires prior therapy' if feats['requires_prior_therapy'] else ''}. "
                       "The model leans mostly on the text, the sponsor's history and the therapeutic area. "
                       "Structured design fields such as phase or sponsor type move the estimate very little.")

# ---------------------------------------------------------------- portfolio
with tab_port:
    f = st.columns(5)
    phases = ["(all)"] + q("select distinct phase_group from marts.fct_trials order by 1").phase_group.tolist()
    areas = ["(all)"] + q("select therapeutic_area from marts.fct_trials group by 1 order by count(*) desc").therapeutic_area.tolist()
    ph = f[0].selectbox("Phase", phases, index=phases.index("Phase 2") if "Phase 2" in phases else 0)
    ar = f[1].selectbox("Therapeutic area", areas, index=areas.index("Neoplasms") if "Neoplasms" in areas else 0)
    status = f[2].selectbox("Status", ["active", "completed", "terminated"], index=0)
    years = q("select min(start_year) a, max(start_year) b from marts.fct_trials where start_year <= year(snapshot_date)").iloc[0]
    yr = f[3].slider("Start year", int(years.a), int(years.b), (max(int(years.a), int(years.b) - 3), int(years.b)))
    sponsor = f[4].text_input("Sponsor contains")
    where, params = ["t.status_group = ?", "t.start_year between ? and ?"], [status, yr[0], yr[1]]
    if ph != "(all)":
        where.append("t.phase_group = ?"); params.append(ph)
    if ar != "(all)":
        where.append("t.therapeutic_area = ?"); params.append(ar)
    if sponsor:
        where.append("t.sponsor_name ilike ?"); params.append(f"%{sponsor}%")
    df = q(f"""select t.nct_id, t.brief_title, t.sponsor_name, t.phase_group, t.therapeutic_area, t.start_date,
                      r.risk_termination, r.risk_accrual, r.p_late, r.expected_slip_days, r.drivers
               from marts.fct_trials t join ml.risk_scores r using (nct_id)
               where {' and '.join(where)} order by r.risk_rank_score desc limit 500""", tuple(params))
    df["top_driver"] = df.drivers.map(lambda s: json.loads(s)[0]["driver"] if s else None)
    st.write(f"{len(df):,} trials, highest risk first")
    st.dataframe(df.drop(columns="drivers"), hide_index=True, width="stretch",
                 column_config={"risk_termination": st.column_config.ProgressColumn("termination risk", min_value=0, max_value=1, format="%.2f"),
                                "risk_accrual": st.column_config.ProgressColumn("accrual risk", min_value=0, max_value=1, format="%.2f"),
                                "p_late": st.column_config.ProgressColumn("P(late)", min_value=0, max_value=1, format="%.2f")})
    wl = config.RESULTS_DIR / "watchlist.md"
    if wl.exists():
        with st.expander("Weekly watchlist (what changed since the last run)"):
            st.markdown(wl.read_text())

# ---------------------------------------------------------------- ask
with tab_ask:
    st.write("Ask in plain English. The agent queries the warehouse and model outputs and cites trial IDs.")
    examples = ["Which Phase 2 oncology trials starting this year are most likely to stall, and why?",
                "Which therapeutic areas had the highest termination rate for trials started 2010-2015?",
                "What are the most common reasons Phase 3 trials were terminated?"]
    from trialscope.llm import llm_configured
    use_llm = llm_configured()
    st.caption("Mode: **LLM agent** (LangGraph + SQL tools)" if use_llm else
               "Mode: **offline**. Answers are computed directly with SQL over the warehouse, with no LLM or API key. "
               "Set `OPENAI_BASE_URL` (a free local Ollama model) or `OPENAI_API_KEY` in `.env` to enable free-text questions.")
    question = st.selectbox("Question", examples, accept_new_options=True)
    if st.button("Ask", type="primary"):
        res = None
        if use_llm:
            try:
                from trialscope.agent import ask
                with st.spinner("Querying…"):
                    res = ask(question)
            except Exception as e:
                st.warning(f"The LLM call failed ({type(e).__name__}); showing the offline answer instead.")
        if res is not None:
            st.markdown(res["answer"])
            st.caption(f"tools: {', '.join(t['name'] for t in res['tool_calls'])} · {res['latency_s']}s · "
                       f"${res['usage']['cost_usd']:.4f}" + (f" · ⚠️ unverified IDs: {res['uncited_ids']}" if res["uncited_ids"] else ""))
            with st.expander("Tool calls"):
                st.json(res["tool_calls"])
        else:
            from trialscope.agent import ask_offline
            off = ask_offline(question)
            if off is None:
                st.info("Offline mode understands questions about the riskiest trials, termination rates by area, and "
                        "termination reasons. Try one of the examples above, or enable an LLM for free-text questions.")
            else:
                st.markdown(off["answer"])
                st.dataframe(off["table"], hide_index=True, width="stretch")

# ---------------------------------------------------------------- insights
with tab_insight:
    p = config.RESULTS_DIR / "insights.md"
    figs = config.RESULTS_DIR / "figures"
    c = st.columns(2)
    for i, name in enumerate(["termination_by_area.png", "km_by_phase.png", "sites_vs_recruitment.png", "km_by_area.png"]):
        if (figs / name).exists():
            c[i % 2].image(str(figs / name))
    if p.exists():
        with st.expander("Full findings report"):
            st.markdown("\n".join(l for l in p.read_text().splitlines() if not l.startswith("![")))
    rs = config.RESULTS_DIR / "termination_reasons.md"
    if rs.exists():
        with st.expander("Termination reasons & topics"):
            st.markdown(rs.read_text())

# ---------------------------------------------------------------- model & data quality
with tab_model:
    cal = config.RESULTS_DIR / "figures" / "calibration.png"
    for name in ("classifier_report.md", "survival_report.md", "drift.md", "extraction_eval.md"):
        p = config.RESULTS_DIR / name
        if p.exists():
            with st.expander(name.replace("_", " ").replace(".md", ""), expanded=name == "classifier_report.md"):
                st.markdown("\n".join(l for l in p.read_text().splitlines() if not l.startswith("![")))
                if name == "classifier_report.md" and cal.exists():
                    st.image(str(cal), width=460)
    st.markdown("**Pipeline loads**")
    if config.WAREHOUSE.exists():
        try:
            st.dataframe(q("select * from raw._loaded_files order by loaded_at desc"), hide_index=True)
        except Exception:
            st.caption("Load history is not included in this copy.")
