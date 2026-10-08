"""Eligibility-criteria text features: rule-based counts/flags + dense text embeddings.

Strict criteria usually mean slow recruitment, so these are some of the most useful
signals that are available at registration time.

    python -m trialscope.text.eligibility           # writes ml.eligibility_features and ml.text_embeddings
"""
from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd

from trialscope import config, db

BULLET = re.compile(r"^\s*(?:[*\-•·▪◦]|\d{1,2}[.)]|[a-z][.)]|[ivx]{1,4}[.)])\s+", re.I)
EXCL_HEAD = re.compile(r"exclusion\s+criteria\s*:?|\bexclusions?\s*:", re.I)
INCL_HEAD = re.compile(r"inclusion\s+criteria\s*:?|\binclusions?\s*:", re.I)
BIOMARKER = re.compile(r"\b(EGFR|HER2|HER-2|ALK|ROS1|BRAF|KRAS|NRAS|PD-?L1|BRCA[12]?|MSI-?H|dMMR|NTRK|FGFR\d?|IDH[12]?|FLT3|"
                       r"PIK3CA|MET|RET|HLA-[A-Z]\*?\d*)\b|\b(mutation|mutated|amplification|rearrangement|fusion|"
                       r"overexpression|biomarker|genotyp\w*)\b", re.I)
PRIOR_THERAPY = re.compile(r"\b(prior|previous(ly)?)\b[^.\n]{0,60}\b(therap\w*|treatment|regimen|line)|\brefractory\b|\brelapsed\b|"
                           r"\bat least (one|two|1|2) (prior )?lines?\b", re.I)
ECOG = re.compile(r"(ECOG|performance status)[^\n]{0,40}?\b([0-4])\s*(?:-|–|to|or)\s*([0-4])\b", re.I)
BRAIN_METS = re.compile(r"(brain|CNS|central nervous system)\s+metasta", re.I)
LAB_VALUE = re.compile(r"(\d+(\.\d+)?\s*(x|×)\s*(ULN|upper limit)|[<>≤≥]=?\s*\d)", re.I)


def split_criteria(text: str | None) -> tuple[list[str], list[str]]:
    """Return (inclusion items, exclusion items). Handles bullets, numbering and plain lines."""
    if not text:
        return [], []
    m = EXCL_HEAD.search(text)
    inc_txt, exc_txt = (text[: m.start()], text[m.end():]) if m else (text, "")
    inc_txt = INCL_HEAD.sub("", inc_txt)

    def items(block: str) -> list[str]:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        bullets = [BULLET.sub("", l).strip() for l in lines if BULLET.match(l)]
        if bullets:
            return [b for b in bullets if len(b) > 2]
        return [l.strip() for l in lines if len(l.strip()) > 15 and not l.strip().endswith(":")]

    return items(inc_txt), items(exc_txt)


def rule_features(text: str | None) -> dict:
    inc, exc = split_criteria(text)
    t = text or ""
    ecog = ECOG.search(t)
    return {
        "n_inclusion": len(inc),
        "n_exclusion": len(exc),
        "elig_chars": len(t),
        "requires_biomarker": bool(any(BIOMARKER.search(x) for x in inc)),
        "requires_prior_therapy": bool(any(PRIOR_THERAPY.search(x) for x in inc)),
        "ecog_max": int(ecog.group(3)) if ecog else None,
        "excludes_brain_mets": bool(any(BRAIN_METS.search(x) for x in exc)),
        "n_lab_thresholds": len(LAB_VALUE.findall(t)),
    }


def trim_exclusions(text: str, keep: int) -> str:
    """Keep the inclusion list and only the first `keep` exclusion criteria (a quick way to try a simpler protocol)."""
    m = EXCL_HEAD.search(text or "")
    if not m:
        return text
    kept, n = [], 0
    for line in text[m.end():].splitlines():
        if BULLET.match(line):
            n += 1
            if n > keep:
                break
        kept.append(line)
    return text[:m.end()] + "\n".join(kept)


def embed_local(texts: pd.Series, fit_mask: np.ndarray, dim: int = config.EMBED_DIM) -> np.ndarray:
    """TF-IDF + truncated SVD, fitted on the TRAINING cohort only (no information from later trials)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True, stop_words="english")
    svd = TruncatedSVD(n_components=dim, random_state=config.SEED)
    svd.fit(vec.fit_transform(texts[fit_mask]))
    with (config.MODEL_DIR / "text_embedder.pkl").open("wb") as f:     # lets the dashboard embed text it has never seen
        pickle.dump({"vec": vec, "svd": svd}, f)
    return _project(vec, svd, texts)


def _project(vec, svd, texts) -> np.ndarray:
    X = svd.transform(vec.transform(texts))
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def embed_text(title: str, area: str, criteria: str) -> np.ndarray:
    """Embed one trial's registration-time text exactly as the pipeline did (needs a pipeline run with EMBED_MODEL=local)."""
    with (config.MODEL_DIR / "text_embedder.pkl").open("rb") as f:
        m = pickle.load(f)
    text = pd.Series([f"{title or ''}. {area or ''}. {criteria or ''}"[:20000]])
    return _project(m["vec"], m["svd"], text)[0]


def embed_openai(texts: pd.Series, dim: int = config.EMBED_DIM) -> np.ndarray:
    from langchain_openai import OpenAIEmbeddings

    emb = OpenAIEmbeddings(model=config.EMBED_MODEL, dimensions=dim, chunk_size=256)
    return np.asarray(emb.embed_documents([t[:8000] for t in texts]), dtype=np.float32)


def build() -> None:
    df = db.query(
        "select s.nct_id, s.start_year, s.brief_title, s.eligibility_criteria, t.therapeutic_area "
        "from staging.stg_studies s join marts.fct_trials t using (nct_id)")
    feats = pd.DataFrame([rule_features(t) for t in df.eligibility_criteria])
    feats.insert(0, "nct_id", df.nct_id.values)
    db.write_table(feats, "eligibility_features")
    print(f"eligibility features for {len(feats):,} trials; median exclusions {feats.n_exclusion.median():.0f}, "
          f"biomarker-gated {feats.requires_biomarker.mean():.1%}")

    # what the embedding sees: title + area + criteria (all registration-time text)
    text = (df.brief_title.fillna("") + ". " + df.therapeutic_area.fillna("") + ". " + df.eligibility_criteria.fillna("")).str[:20000]
    if config.EMBED_MODEL == "local":
        X = embed_local(text, (df.start_year <= config.TRAIN_END_YEAR).fillna(False).to_numpy())
    else:
        X = embed_openai(text)
    emb = pd.DataFrame(X, columns=[f"e{i}" for i in range(X.shape[1])])
    emb.insert(0, "nct_id", df.nct_id.values)
    db.write_table(emb, "text_embeddings")
    print(f"text embeddings: {X.shape} via {config.EMBED_MODEL}")


if __name__ == "__main__":
    build()
