"""Why do trials stop? Categorise the free-text `why_stopped` field and cluster it into topics.

1. Rules: ordered keyword patterns -> {enrollment, funding, business, safety, efficacy, logistics,
   regulatory, covid, other}. Transparent and free.
2. Optional LLM classifier (--llm) for texts the rules can't place, or all texts (--llm-all).
3. Unsupervised topics: TF-IDF + k-means, top terms per cluster, cross-tabbed with the rule
   categories — shows what the hand-made taxonomy misses.
Evaluated against labels/reason_labels.csv if you've labelled some (python -m trialscope.text.label reasons).

    python -m trialscope.text.termination [--llm] [--k 10]
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import pandas as pd

from trialscope import config, db

CATEGORIES = ["enrollment", "funding", "business", "safety", "efficacy", "logistics", "regulatory", "covid", "other"]
RULES = [  # first match wins; order matters (e.g. "slow accrual due to COVID" -> covid)
    ("covid", r"covid|pandemic|sars[- ]?cov|coronavirus"),
    ("safety", r"safety|toxicit|adverse|side[- ]effect|\bdeath|\bdied|serious|harm|dsmb.*(safety|adverse)"),
    ("efficacy", r"futil|efficacy|lack of (effect|benefit|response)|no (benefit|difference)|interim analysis|ineffective|endpoint"),
    ("enrollment", r"accru|enrol|recruit|eligible (patients|participants|subjects)|participants? (were|was) (not )?(enrolled|found)|"
                   r"(low|slow|poor|insufficient|inadequate) (participation|numbers?|patient)|no (patients|participants|subjects)|"
                   r"unable to (find|identify|recruit)"),
    ("funding", r"fund|financ|budget|grant|money|cost"),
    ("business", r"business|strateg|sponsor('s)? decision|company decision|portfolio|commercial|priorit|acqui|merger|"
                 r"development program|discontinu\w+ (the )?(development|program)"),
    ("regulatory", r"\bfda\b|regulat|\birb\b|ethic|approval|clinical hold|license"),
    ("logistics", r"\bpi\b|principal investigator|investigator (left|moved|relocat)|left the institution|relocat|"
                  r"drug supply|supply|availability|staff|personnel|site (closed|closure)|logistic|manufactur"),
]
COMPILED = [(c, re.compile(p, re.I)) for c, p in RULES]
# "Business decision, not related to safety" is very common: drop negated mentions before matching
_TERM = r"(safety|efficacy|adverse( events?)?|toxicit\w*)"
NEGATED = re.compile(rf"\b(not|no|unrelated|without)\b[^.;,]{{0,25}}?\b{_TERM}(\s*(or|and|/|,)\s*{_TERM})*"
                     r"(\s+(concerns?|issues?|reasons?|related))?", re.I)


def classify_rule(text: str | None) -> str:
    if not text:
        return "other"
    text = NEGATED.sub(" ", text)
    for cat, rx in COMPILED:
        if rx.search(text):
            return cat
    return "other"


def classify_llm(texts: list[str], workers: int = 8) -> list[str]:
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from trialscope.llm import get_chat

    class Reason(BaseModel):
        category: Literal[tuple(CATEGORIES)] = Field(description="primary reason the trial stopped")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify why a clinical trial stopped. Categories: enrollment (accrual/recruitment problems), "
                   "funding, business (sponsor/strategic/portfolio decisions), safety, efficacy (futility/lack of effect), "
                   "logistics (PI left, drug supply, staffing, site issues), regulatory, covid, other. Pick the primary cause."),
        ("human", "{text}")])
    chain = prompt | get_chat().with_structured_output(Reason, method="function_calling")
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: chain.invoke({"text": t}).category, texts))


def topics(texts: pd.Series, k: int) -> tuple[np.ndarray, pd.DataFrame]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(texts.str.lower())
    km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(X)
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for c in range(k):
        top = terms[np.argsort(-km.cluster_centers_[c])[:8]]
        rows.append({"cluster": c, "size": int((km.labels_ == c).sum()), "top_terms": ", ".join(top)})
    return km.labels_, pd.DataFrame(rows).sort_values("size", ascending=False)


def evaluate(df: pd.DataFrame) -> dict | None:
    p = config.LABELS_DIR / "reason_labels.csv"
    if not p.exists():
        return None
    gold = pd.read_csv(p).merge(df, on="nct_id")
    out = {"n_labelled": len(gold), "rule_accuracy": round(float((gold.reason_rule == gold.label).mean()), 4)}
    if "reason_llm" in gold and gold.reason_llm.notna().any():
        g = gold[gold.reason_llm.notna()]
        out["llm_accuracy"] = round(float((g.reason_llm == g.label).mean()), 4)
    return out


def build(llm: bool = False, llm_all: bool = False, k: int = 10) -> pd.DataFrame:
    df = db.query("select nct_id, why_stopped, status_group, phase_group, therapeutic_area, sponsor_class "
                  "from marts.fct_trials where why_stopped is not null and status_group in ('terminated','withdrawn','suspended')")
    df["reason_rule"] = df.why_stopped.map(classify_rule)
    df["reason_llm"] = None
    if llm or llm_all:
        sel = df.index if llm_all else df.index[df.reason_rule == "other"]
        print(f"LLM-classifying {len(sel):,} reasons with {config.CHAT_MODEL} ...")
        df.loc[sel, "reason_llm"] = classify_llm(df.loc[sel, "why_stopped"].tolist())
    df["reason"] = df.reason_llm.fillna(df.reason_rule)
    labels, top = topics(df.why_stopped, k=min(k, max(2, len(df) // 20)))
    df["topic"] = labels
    db.write_table(df, "termination_reasons")

    ct = pd.crosstab(df.topic, df.reason)
    lines = ["# Termination reasons", "", f"{len(df):,} stopped trials with a stated reason.", "",
             "## Rule-based categories", "", df.reason.value_counts(normalize=True).round(3).to_frame("share").to_markdown(), "",
             "## Topics (TF-IDF + k-means)", "", top.to_markdown(index=False), "",
             "## Topic x category", "", ct.to_markdown()]
    ev = evaluate(df)
    if ev:
        lines += ["", "## Accuracy vs hand labels", "", str(ev)]
    (config.RESULTS_DIR / "termination_reasons.md").write_text("\n".join(lines) + "\n")
    print(df.reason.value_counts().to_string())
    print(f"'other' (unplaced by rules): {(df.reason_rule == 'other').mean():.1%}")
    if ev:
        print("vs hand labels:", ev)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="LLM-classify only what the rules call 'other'")
    ap.add_argument("--llm-all", action="store_true", help="LLM-classify everything (compare with rules)")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    build(a.llm, a.llm_all, a.k)
