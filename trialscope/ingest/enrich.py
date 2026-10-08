"""Optional enrichment from PubMed (NCBI E-utilities) and openFDA (Drugs@FDA).

pubmed   Which trials have publications that cite their NCT number (PubMed "secondary source ID")?
         An OUTCOME signal (did results come out?), used in analysis — never as a model feature.
         Batched: one esearch per 200 NCT IDs (POST), then efetch to map PMIDs back to trials.
fda      First FDA approval date of each drug name (Drugs@FDA). Feature `drug_previously_approved`:
         any DRUG intervention approved BEFORE the trial's start date -> leakage-safe.

    python -m trialscope.ingest.enrich pubmed [--limit 20000]
    python -m trialscope.ingest.enrich fda [--top 3000]
"""
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

from trialscope import config, db

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENFDA = "https://api.fda.gov/drug/drugsfda.json"
HEADERS = {"User-Agent": "TrialScope portfolio project"}


def _sleep_ncbi():
    time.sleep(0.11 if config.NCBI_API_KEY else 0.35)   # 10 req/s with key, 3 req/s without


def _post(session, url, data, tries=5):
    for i in range(tries):
        try:
            r = session.post(url, data=data, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503):
                r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"failed: {url}")


def parse_efetch(xml_text: str, wanted: set[str]) -> list[dict]:
    rows = []
    for art in ET.fromstring(xml_text).iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID")
        year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        for db_ in art.iter("DataBank"):
            if (db_.findtext("DataBankName") or "").lower().startswith("clinicaltrials"):
                for acc in db_.iter("AccessionNumber"):
                    nct = (acc.text or "").strip().upper()
                    if nct in wanted:
                        rows.append({"nct_id": nct, "pmid": pmid, "pub_year": int(year) if year.isdigit() else None})
    return rows


def pubmed(limit: int | None = None, batch: int = 200) -> pd.DataFrame:
    ncts = db.query(f"select nct_id from marts.fct_trials where is_resolved and start_year between {config.MODEL_MIN_START_YEAR} "
                    f"and {config.TEST_END_YEAR} order by nct_id").nct_id.tolist()[:limit]
    s, rows = requests.Session(), []
    key = {"api_key": config.NCBI_API_KEY} if config.NCBI_API_KEY else {}
    for i in tqdm(range(0, len(ncts), batch), desc="PubMed"):
        chunk = ncts[i: i + batch]
        term = " OR ".join(f"{n}[si]" for n in chunk)
        r = _post(s, f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": term, "retmax": 10000, "retmode": "json", **key})
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        _sleep_ncbi()
        for j in range(0, len(ids), 200):
            x = _post(s, f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(ids[j: j + 200]), "retmode": "xml", **key})
            rows += parse_efetch(x.text, set(chunk))
            _sleep_ncbi()
    out = pd.DataFrame(rows, columns=["nct_id", "pmid", "pub_year"]).drop_duplicates()
    db.write_table(out, "pubmed_links")
    print(f"{out.nct_id.nunique():,} of {len(ncts):,} trials have >= 1 PubMed article citing them")
    return out


def first_approval(results: list[dict]) -> tuple[str | None, str | None]:
    best = (None, None)
    for app in results:
        for sub in app.get("submissions", []):
            if sub.get("submission_type") == "ORIG" and sub.get("submission_status") == "AP":
                d = sub.get("submission_status_date")
                if d and (best[0] is None or d < best[0]):
                    best = (d, app.get("application_number"))
    return best


def fda(top: int = 3000) -> pd.DataFrame:
    names = db.query("select intervention_name as name, count(distinct nct_id) n from staging.stg_interventions "
                     "where intervention_type in ('DRUG','BIOLOGICAL') and regexp_matches(intervention_name, '^[a-z][a-z -]{3,40}$') "
                     "and intervention_name not in ('placebo','saline','normal saline','standard of care','vehicle') "
                     "group by 1 order by 2 desc limit ?", [top])
    s, rows = requests.Session(), []
    key = {"api_key": config.OPENFDA_API_KEY} if config.OPENFDA_API_KEY else {}
    for name in tqdm(names.name, desc="openFDA"):
        q = f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"'
        for i in range(4):
            r = s.get(OPENFDA, params={"search": q, "limit": 100, **key}, headers=HEADERS, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(5 * (i + 1))
        d, appno = first_approval(r.json().get("results", [])) if r.status_code == 200 else (None, None)
        rows.append({"drug_name": name, "first_approval_date": datetime.strptime(d, "%Y%m%d").date() if d else None,
                     "application_number": appno})
        time.sleep(0.05 if key else 0.26)   # ~240 req/min without a key
    out = pd.DataFrame(rows)
    db.write_table(out, "fda_approvals")
    link = db.query("""
        select i.nct_id,
               bool_or(a.first_approval_date is not null and a.first_approval_date < s.start_date) as drug_previously_approved
        from staging.stg_interventions i
        join staging.stg_studies s using (nct_id)
        left join ml.fda_approvals a on a.drug_name = i.intervention_name
        where i.intervention_type in ('DRUG','BIOLOGICAL') group by 1""")
    db.write_table(link, "trial_drug_approval")
    print(f"{out.first_approval_date.notna().sum():,} of {len(out):,} drug names matched an FDA approval; "
          f"{link.drug_previously_approved.mean():.1%} of drug trials test a previously approved drug")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["pubmed", "fda"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--top", type=int, default=3000)
    a = ap.parse_args()
    pubmed(a.limit) if a.source == "pubmed" else fda(a.top)
