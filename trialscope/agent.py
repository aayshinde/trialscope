"""Analyst agent: answers portfolio questions by querying the warehouse and model outputs.

Built with LangGraph (agent <-> tools loop). Tools:
  run_sql            read-only SELECT over marts.* and ml.* (guarded, row-limited)
  trial_profile      one trial: attributes, risk scores, top drivers, delay prediction
  similar_trials     how the most similar earlier trials ended
  risk_leaderboard   highest-risk active/not-yet-started trials for a filter
Every NCT ID in the final answer is checked against the tool outputs (citation check).

    python -m trialscope.agent "Which Phase 2 oncology trials starting this year are most likely to stall, and why?"
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from functools import lru_cache
from typing import Optional

import duckdb
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from trialscope import config
from trialscope.llm import UsageTracker, get_chat

NCT_RE = re.compile(r"\bNCT\d{8}\b", re.I)
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|"
                       r"call|set|checkpoint|vacuum|grant|revoke)\b|read_(csv|parquet|json)|\bglob\s*\(", re.I)
ALLOWED_SCHEMAS = ("marts", "ml")
MAX_ROWS = 50

TABLE_DOCS = {
    "marts.fct_trials": "one row per trial: nct_id, brief_title, status_group (completed/terminated/withdrawn/suspended/active/"
                        "unknown), overall_status, why_stopped, start_date, start_year, phase_group ('Phase 1','Phase 1/2','Phase 2',"
                        "'Phase 2/3','Phase 3','Phase 4','Not applicable'), therapeutic_area (MeSH branch, e.g. 'Neoplasms' = oncology), "
                        "is_rare_disease, sponsor_name, sponsor_class (INDUSTRY/NIH/OTHER...), enrollment_count, n_locations, "
                        "terminated_flag (1 terminated, 0 completed, null unresolved), duration_days, planned_primary_completion, "
                        "is_overdue, sponsor_prior_term_rate, n_status_changes, n_completion_slips",
    "ml.risk_scores": "model outputs per trial: nct_id, risk_termination (calibrated probability), risk_accrual, "
                      "risk_percentile_active, p_late (prob. of missing planned primary completion), predicted_completion, "
                      "expected_slip_days, drivers (JSON list of top SHAP drivers)",
    "ml.termination_reasons": "nct_id, why_stopped, reason (enrollment/funding/business/safety/efficacy/logistics/regulatory/covid/other), topic",
    "marts.dim_sponsors": "sponsor_key, sponsor_name, sponsor_class, n_trials, n_terminated, n_completed, termination_rate",
    "marts.mart_area_phase_stats": "therapeutic_area, phase_group, n_resolved, termination_rate, median_days_to_completion",
    "ml.similar_trials": "nct_id, rank, similar_nct_id, similarity, similar_status, similar_why_stopped, similar_duration_months",
}

SYSTEM = """You are TrialScope, an analyst for clinical-trial portfolio risk. Answer using ONLY data returned by your tools.

Tables (DuckDB SQL; use run_sql for anything the other tools don't cover):
{tables}

Rules:
- Oncology = therapeutic_area 'Neoplasms'. "This year" = {this_year}. Data snapshot date: {snapshot}.
- Risk scores are model estimates (probabilities), not facts — say so, and explain them with the drivers.
- Cite trials by NCT ID, and only IDs that appear in tool results. Never invent IDs, numbers or reasons.
- Prefer risk_leaderboard for "which trials are riskiest" questions, then trial_profile for the why.
- Definitions: "terminated" means status_group = 'terminated' only (withdrawn and suspended are different statuses). "Resolved" means
  terminated_flag is not null (completed or terminated), so a termination rate among resolved trials is avg(terminated_flag) over
  those rows, times 100 for a percentage.
- Pass phase and therapeutic-area values to tools exactly as the user wrote them. The first row of risk_leaderboard is the single highest-risk trial.
- Keep answers concise: a short summary, then a list of trials with the key numbers and reasons."""


# ------------------------------------------------------------------ tools
@lru_cache(maxsize=1)
def _con():
    from trialscope.db import connect

    return connect(read_only=True)


def _df_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object and c != "drivers":      # drivers is JSON: truncating it corrupts it
            df[c] = df[c].astype(str).str.slice(0, 160)
    return df.to_csv(index=False)


def check_sql(sql: str) -> str:
    q = sql.strip().rstrip(";").strip()
    if ";" in q:
        raise ValueError("only one statement allowed")
    if not re.match(r"^(select|with)\b", q, re.I):
        raise ValueError("only SELECT / WITH queries are allowed")
    if FORBIDDEN.search(q):
        raise ValueError("query contains a forbidden keyword")
    tables = {t.lower() for t in re.findall(r"\b(?:from|join)\s+([a-z_]+\.[a-z_]+|[a-z_]+)", q, re.I)}
    bad = [t for t in tables if "." in t and t.split(".")[0] not in ALLOWED_SCHEMAS]
    if bad:
        raise ValueError(f"tables outside {ALLOWED_SCHEMAS}: {bad}")
    return q


def run_sql(query: str) -> str:
    """Run a read-only SQL query (DuckDB dialect) against marts.* and ml.* tables. Returns up to 50 rows as CSV."""
    try:
        q = check_sql(query)
        df = _con().execute(f"select * from ({q}) as _q limit {MAX_ROWS}").df()
        return _df_text(df)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def trial_profile(nct_id: str) -> str:
    """Everything about one trial: design, sponsor, status, risk scores with top drivers, delay prediction."""
    df = _con().execute("""
        select t.nct_id, t.brief_title, t.status_group, t.overall_status, t.why_stopped, t.start_date, t.phase_group,
               t.therapeutic_area, t.is_rare_disease, t.sponsor_name, t.sponsor_class, t.sponsor_prior_term_rate,
               t.planned_primary_completion, t.is_overdue, t.n_completion_slips,
               r.risk_termination, r.risk_accrual, r.risk_percentile_active, r.p_late, r.predicted_completion,
               r.expected_slip_days, r.drivers
        from marts.fct_trials t left join ml.risk_scores r using (nct_id) where t.nct_id = ?""", [nct_id.upper()]).df()
    if df.empty:
        return f"No trial {nct_id} in the warehouse."
    return json.dumps(df.iloc[0].to_dict(), default=str)


def similar_trials(nct_id: str, k: int = 5) -> str:
    """The k most similar trials that started BEFORE this one, and how they ended."""
    from trialscope.models.score import similar_for

    try:
        return _df_text(similar_for(nct_id, k))
    except Exception as e:
        return f"ERROR: {e}"


def risk_leaderboard(phase_group: Optional[str] = None, therapeutic_area: Optional[str] = None,
                     start_year_from: Optional[int] = None, start_year_to: Optional[int] = None,
                     status_group: str = "active", sponsor_class: Optional[str] = None, n: int = 10) -> str:
    """Highest termination-risk trials for a filter (default: active trials), with drivers and delay risk."""
    where, params = ["t.status_group = ?"], [status_group]
    for col, val in (("t.phase_group", phase_group), ("t.therapeutic_area", therapeutic_area), ("t.sponsor_class", sponsor_class)):
        if val:
            where.append(f"lower({col}) = lower(?)")        # small LLMs often change capitalisation; match case-insensitively
            params.append(val)
    if start_year_from:
        where.append("t.start_year >= ?")
        params.append(start_year_from)
    if start_year_to:
        where.append("t.start_year <= ?")
        params.append(start_year_to)
    df = _con().execute(f"""
        select t.nct_id, t.brief_title, t.phase_group, t.therapeutic_area, t.sponsor_name, t.start_date,
               round(r.risk_termination, 3) risk_termination, round(r.risk_accrual, 3) risk_accrual,
               round(r.p_late, 3) p_late, r.expected_slip_days, r.drivers
        from marts.fct_trials t join ml.risk_scores r using (nct_id)
        where {' and '.join(where)} order by r.risk_rank_score desc limit {min(int(n), 25)}""", params).df()
    return _df_text(df)


class _Lb(BaseModel):
    phase_group: Optional[str] = Field(None, description="'Phase 1','Phase 1/2','Phase 2','Phase 2/3','Phase 3','Phase 4','Not applicable'")
    therapeutic_area: Optional[str] = Field(None, description="MeSH branch name, e.g. 'Neoplasms'")
    start_year_from: Optional[int] = None
    start_year_to: Optional[int] = None
    status_group: str = Field("active", description="active / completed / terminated ...")
    sponsor_class: Optional[str] = None
    n: int = 10


TOOLS = [
    StructuredTool.from_function(run_sql),
    StructuredTool.from_function(trial_profile),
    StructuredTool.from_function(similar_trials),
    StructuredTool.from_function(risk_leaderboard, args_schema=_Lb),
]


# ------------------------------------------------------------------ graph
def system_prompt() -> str:
    snap = _con().execute("select max(snapshot_date) from marts.fct_trials").fetchone()[0]
    return SYSTEM.format(tables="\n".join(f"- {k}: {v}" for k, v in TABLE_DOCS.items()),
                         this_year=snap.year if snap else time.localtime().tm_year, snapshot=snap)


def build_agent(llm=None):
    bound = (llm or get_chat()).bind_tools(TOOLS)
    sys_msg = SystemMessage(system_prompt())

    def agent(state: MessagesState, config=None):
        return {"messages": [bound.invoke([sys_msg] + state["messages"], config)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


def ask(question: str, llm=None, app=None) -> dict:
    app = app or build_agent(llm)
    tracker, t0 = UsageTracker(), time.time()
    state = app.invoke({"messages": [HumanMessage(question)]}, {"callbacks": [tracker], "recursion_limit": 16})
    msgs = state["messages"]
    tool_text = " ".join(m.content for m in msgs if isinstance(m, ToolMessage))
    answer = msgs[-1].content if isinstance(msgs[-1].content, str) else str(msgs[-1].content)
    cited = sorted({c.upper() for c in NCT_RE.findall(answer)})
    seen = {c.upper() for c in NCT_RE.findall(tool_text)}
    return {
        "answer": answer,
        "tool_calls": [{"name": tc["name"], "args": tc["args"]} for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls],
        "cited_trials": cited,
        "uncited_ids": [c for c in cited if c not in seen],     # IDs in the answer that no tool returned -> hallucinated
        "latency_s": round(time.time() - t0, 2),
        "usage": tracker.summary(),
    }


def ask_offline(question: str) -> Optional[dict]:
    """No-LLM fallback: routes the common question shapes to the same SQL the agent would run. None if not recognised."""
    ql = question.lower()
    phase = re.search(r"phase\s*(\d(?:/\d)?)", ql)
    phase = f"Phase {phase.group(1)}" if phase else None
    area = "Neoplasms" if re.search(r"oncolog|cancer|neoplasm|tumou?r", ql) else None
    snap = _con().execute("select max(snapshot_date) from marts.fct_trials").fetchone()[0]

    if re.search(r"stall|riskiest|highest.risk|most likely|at risk", ql):
        year = snap.year if "this year" in ql and snap else None
        df = pd.read_csv(io.StringIO(risk_leaderboard(phase, area, year, None, "active", None, 10)))
        if df.empty and year:     # few trials start in the snapshot year; widen to the last 2 years
            df = pd.read_csv(io.StringIO(risk_leaderboard(phase, area, year - 1, None, "active", None, 10)))
        if df.empty:
            return {"answer": "No active trials match that filter.", "table": df}
        df["top_driver"] = df.pop("drivers").map(lambda s: json.loads(s)[0]["driver"] if isinstance(s, str) and s else None)
        scope = " ".join(x for x in [phase, "oncology" if area else None, "active trials"] if x)
        return {"answer": f"**Highest-risk {scope}**, ranked by the model's calibrated termination risk. "
                          f"`top_driver` is the largest SHAP contribution to each score.", "table": df}

    if "termination rate" in ql or ("area" in ql and "terminat" in ql):
        yrs = re.search(r"(20\d\d)\D{1,6}(20\d\d)", ql)
        lo, hi = (int(yrs.group(1)), int(yrs.group(2))) if yrs else (2008, 2019)
        df = _con().execute("""select therapeutic_area, count(*) as n_resolved, round(avg(terminated_flag), 3) as termination_rate
                               from marts.fct_trials where terminated_flag is not null and start_year between ? and ?
                               group by 1 having count(*) >= 30 order by 3 desc limit 10""", [lo, hi]).df()
        return {"answer": f"**Termination rate by therapeutic area**, trials started {lo}-{hi} (areas with at least 30 resolved trials).",
                "table": df}

    if "reason" in ql and "terminat" in ql:
        df = _con().execute(f"""select r.reason, count(*) as n_trials, round(count(*) / sum(count(*)) over (), 3) as share
                                from ml.termination_reasons r join marts.fct_trials t using (nct_id)
                                {'where t.phase_group = ?' if phase else ''} group by 1 order by 2 desc""",
                            [phase] if phase else []).df()
        return {"answer": f"**Why {phase or 'trials'} were terminated**, classified from the free-text stop reason.", "table": df}
    return None


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which Phase 2 oncology trials starting this year are most likely to stall, and why?"
    r = ask(q)
    print(r["answer"])
    print("\n--- tools:", [t["name"] for t in r["tool_calls"]], "| cited:", r["cited_trials"],
          "| unverified IDs:", r["uncited_ids"], "| cost $", r["usage"]["cost_usd"])
