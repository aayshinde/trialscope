"""Synthetic studies in the exact ClinicalTrials.gov v2 JSON shape.

ONLY for offline tests and a quick demo of the pipeline before the real download.
Never report results computed on this data — the real project runs on the live API.

    python -m trialscope.demo_data --n 6000          # writes data/landing/demo_full_*.jsonl.gz
    python -m trialscope.demo_data --update          # a later "daily update" batch (status changes, date slips)
"""
from __future__ import annotations

import argparse
import gzip
import json
import zlib
from datetime import date, timedelta

import numpy as np

from trialscope import config

SNAPSHOT = date(2026, 10, 1)

AREAS = [  # (browse branch name, weight, base termination log-odds, duration multiplier)
    ("Neoplasms", 0.30, 0.35, 1.3), ("Heart and Blood Diseases", 0.10, 0.0, 1.0),
    ("Nervous System Diseases", 0.10, 0.25, 1.1), ("Infections", 0.10, -0.1, 0.8),
    ("Nutritional and Metabolic Diseases", 0.10, -0.2, 0.9), ("Behaviors and Mental Disorders", 0.08, 0.1, 1.0),
    ("Respiratory Tract (Lung and Bronchial) Diseases", 0.07, 0.0, 0.9), ("Immune System Diseases", 0.08, 0.05, 1.0),
    ("Skin and Connective Tissue Diseases", 0.07, -0.1, 0.8),
]
CONDITIONS = {
    "Neoplasms": ["Non-Small Cell Lung Cancer", "Breast Cancer", "Acute Myeloid Leukemia", "Glioblastoma", "Pancreatic Cancer"],
    "Heart and Blood Diseases": ["Heart Failure", "Atrial Fibrillation", "Hypertension"],
    "Nervous System Diseases": ["Alzheimer Disease", "Parkinson Disease", "Multiple Sclerosis", "Migraine"],
    "Infections": ["HIV Infections", "COVID-19", "Hepatitis C"],
    "Nutritional and Metabolic Diseases": ["Type 2 Diabetes", "Obesity", "Fabry Disease"],
    "Behaviors and Mental Disorders": ["Major Depressive Disorder", "Schizophrenia", "Opioid Use Disorder"],
    "Respiratory Tract (Lung and Bronchial) Diseases": ["Asthma", "COPD", "Idiopathic Pulmonary Fibrosis"],
    "Immune System Diseases": ["Rheumatoid Arthritis", "Systemic Lupus Erythematosus", "Psoriasis"],
    "Skin and Connective Tissue Diseases": ["Atopic Dermatitis", "Psoriasis", "Systemic Sclerosis"],
}
RARE = {"Glioblastoma", "Fabry Disease", "Idiopathic Pulmonary Fibrosis", "Systemic Sclerosis", "Acute Myeloid Leukemia"}
DRUGS = ["pembrolizumab", "nivolumab", "metformin", "semaglutide", "atorvastatin", "adalimumab", "remdesivir",
         "osimertinib", "olaparib", "donepezil", "sertraline", "dupilumab", "TS-1042", "ABX-221", "XR-907"]

SPONSORS = [  # (canonical, surface variants, class, frailty)
    ("Pfizer", ["Pfizer", "Pfizer Inc.", "Pfizer, Inc", "PFIZER INC"], "INDUSTRY", -0.3),
    ("Hoffmann-La Roche", ["Hoffmann-La Roche", "Roche", "F. Hoffmann-La Roche Ltd"], "INDUSTRY", -0.2),
    ("Novartis", ["Novartis Pharmaceuticals", "Novartis", "Novartis Pharma AG"], "INDUSTRY", -0.2),
    ("AstraZeneca", ["AstraZeneca", "AstraZeneca AB"], "INDUSTRY", -0.1),
    ("Merck Sharp & Dohme", ["Merck Sharp & Dohme LLC", "Merck Sharp & Dohme Corp.", "Merck Sharp and Dohme"], "INDUSTRY", -0.25),
    ("Eli Lilly", ["Eli Lilly and Company", "Eli Lilly & Co"], "INDUSTRY", -0.2),
    ("Bristol-Myers Squibb", ["Bristol-Myers Squibb", "Bristol Myers Squibb"], "INDUSTRY", 0.0),
    ("National Cancer Institute", ["National Cancer Institute (NCI)", "National Cancer Institute"], "NIH", 0.1),
    ("M.D. Anderson Cancer Center", ["M.D. Anderson Cancer Center", "MD Anderson Cancer Center"], "OTHER", 0.35),
    ("Mayo Clinic", ["Mayo Clinic", "Mayo Clinic, Rochester"], "OTHER", 0.2),
    ("University of California, San Francisco", ["University of California, San Francisco", "UCSF"], "OTHER", 0.3),
] + [(f"{n} Therapeutics", [f"{n} Therapeutics, Inc.", f"{n} Therapeutics Inc", f"{n} Therapeutics"], "INDUSTRY", f)
     for n, f in [("Acme", 0.9), ("Borealis", 0.7), ("Cygnus", 1.1), ("Delphi", 0.5), ("Equinox", 0.8), ("Fathom", 1.0)]]
SPONSOR_W = np.array([8, 6, 6, 5, 6, 5, 4, 7, 6, 5, 4, 2, 2, 2, 2, 2, 2], float)

REASONS = {
    "enrollment": ["Slow accrual", "Low enrollment", "Poor recruitment", "The study was stopped due to slow enrollment",
                   "Insufficient number of eligible participants", "Unable to recruit subjects"],
    "funding": ["Lack of funding", "Funding ended", "Sponsor withdrew financial support"],
    "business": ["Business decision", "Sponsor decision", "Strategic portfolio prioritization", "Company decision not related to safety"],
    "safety": ["Safety concerns", "Unacceptable toxicity", "Stopped by DSMB due to adverse events"],
    "efficacy": ["Futility at interim analysis", "Lack of efficacy", "Interim analysis showed no benefit"],
    "logistics": ["PI left the institution", "Drug supply issues", "Investigator relocated"],
}
BIOMARKERS = ["EGFR mutation", "HER2-positive disease", "PD-L1 expression of at least 50%", "BRCA1/2 mutation",
              "KRAS G12C mutation", "ALK rearrangement"]
INCL = ["Age >= 18 years", "Signed informed consent", "Adequate organ function", "Life expectancy of at least 12 weeks",
        "Measurable disease per RECIST 1.1", "ECOG performance status 0-1", "Body mass index between 18 and 40",
        "Confirmed diagnosis by a physician", "Willing to use contraception", "HbA1c between 7.0% and 10.5%",
        "Stable medication for at least 4 weeks", "At least one prior line of therapy"]
EXCL = ["Pregnant or breastfeeding", "Known brain metastases", "Active hepatitis B or C", "History of HIV infection",
        "Prior treatment with an immune checkpoint inhibitor", "Myocardial infarction within 6 months",
        "Uncontrolled hypertension", "eGFR < 30 mL/min", "Participation in another trial within 30 days",
        "Known hypersensitivity to study drug", "Major surgery within 4 weeks", "QTc > 470 ms", "Active autoimmune disease",
        "Systemic corticosteroids > 10 mg/day prednisone", "History of another malignancy within 3 years",
        "Uncontrolled diabetes", "Severe hepatic impairment", "Alcohol or substance abuse", "Psychiatric illness limiting compliance",
        "Prior organ transplant", "Positive SARS-CoV-2 test", "Live vaccine within 30 days"]
PHASES = [(["EARLY_PHASE1"], .03), (["PHASE1"], .18), (["PHASE1", "PHASE2"], .07), (["PHASE2"], .27), (["PHASE3"], .17),
          (["PHASE4"], .10), ([], .18)]
PHASE_EFFECT = {"EARLY_PHASE1": 0.1, "PHASE1": -0.2, "PHASE1|PHASE2": 0.3, "PHASE2": 0.35, "PHASE3": 0.1, "PHASE4": -0.1, "": 0.0}
COUNTRIES = ["United States", "Canada", "Germany", "France", "United Kingdom", "Spain", "China", "Japan", "Korea, Republic of", "Australia"]


def _fmt(d: date, rng) -> str:
    return d.strftime("%Y-%m") if rng.random() < 0.35 else d.isoformat()


def make_study(i: int, rng: np.random.Generator) -> dict:
    nct = f"NCT0{3000000 + i:07d}"
    area_i = rng.choice(len(AREAS), p=np.array([a[1] for a in AREAS]) / sum(a[1] for a in AREAS))
    area, _, area_eff, dur_mult = AREAS[area_i]
    cond = str(rng.choice(CONDITIONS[area]))
    rare = cond in RARE
    ph = PHASES[rng.choice(len(PHASES), p=[p for _, p in PHASES])][0]
    ph_key = "|".join(ph)
    si = rng.choice(len(SPONSORS), p=SPONSOR_W / SPONSOR_W.sum())
    s_canon, s_vars, s_class, frailty = SPONSORS[si]
    sponsor = str(rng.choice(s_vars))
    start = date(2003, 1, 1) + timedelta(days=int(rng.integers(0, (date(2025, 12, 31) - date(2003, 1, 1)).days)))
    first_submit = start - timedelta(days=int(rng.integers(-60, 240)))

    # eligibility text (strictness drives recruitment difficulty)
    n_inc, n_exc = int(rng.integers(3, 13)), int(rng.integers(3, 23))
    biomarker = area == "Neoplasms" and rng.random() < 0.45
    inc = list(rng.choice(INCL, size=min(n_inc, len(INCL)), replace=False))
    if biomarker:
        inc.insert(0, f"Documented {rng.choice(BIOMARKERS)}")
    exc = list(rng.choice(EXCL, size=min(n_exc, len(EXCL)), replace=False))
    elig = "Inclusion Criteria:\n\n" + "\n".join(f"* {x}" for x in inc) + "\n\nExclusion Criteria:\n\n" + "\n".join(f"* {x}" for x in exc)
    strict = (len(exc) - 12) / 6 + (0.8 if biomarker else 0) + (0.6 if rare else 0)

    planned_n = int(np.clip(rng.lognormal({"PHASE1": 3.3, "PHASE3": 6.0, "PHASE4": 5.0}.get(ph_key, 4.3), 0.7), 8, 5000))
    n_sites_planned = int(np.clip(rng.lognormal(3.0 if s_class == "INDUSTRY" and ph_key in ("PHASE3", "PHASE2") else 0.6, 1.0), 1, 600))
    planned_dur = int(365 * np.clip(rng.lognormal(np.log(2.2 * dur_mult), 0.4), 0.4, 8))

    # latent risk -> outcome
    logit = -1.9 + area_eff + PHASE_EFFECT[ph_key] + frailty + 0.35 * strict + (0.4 if s_class == "OTHER" and n_sites_planned < 3 else 0)
    p_term = 1 / (1 + np.exp(-logit))
    true_dur = int(planned_dur * np.exp(rng.normal(0.15 + 0.12 * strict + (0.2 if rare else 0), 0.35)))
    status, why, enroll_n, enroll_type = "", None, planned_n, "ESTIMATED"
    pc_date, pc_type, n_sites = start + timedelta(days=planned_dur), "ESTIMATED", n_sites_planned
    end = start + timedelta(days=true_dur)
    reason = None
    if start > SNAPSHOT:
        status = "NOT_YET_RECRUITING"
    elif rng.random() < p_term and rng.random() < 0.92:
        t_end = start + timedelta(days=int(true_dur * rng.uniform(0.15, 0.8)))
        if t_end <= SNAPSHOT:
            w = np.array([1.2 + max(strict, 0), 0.6 + (0.8 if s_class != "INDUSTRY" else 0), 0.9 if s_class == "INDUSTRY" else 0.2,
                          0.25 + (0.2 if ph_key == "PHASE1" else 0), 0.4 if ph_key in ("PHASE2", "PHASE3") else 0.1, 0.25])
            reason = list(REASONS)[rng.choice(6, p=w / w.sum())]
            status, why = "TERMINATED", str(rng.choice(REASONS[reason]))
            enroll_n, enroll_type = int(planned_n * rng.uniform(0.05, 0.6)), "ACTUAL"
            pc_date, pc_type = t_end, "ACTUAL"
            n_sites = max(1, int(n_sites_planned * rng.uniform(0.2, 0.8)))   # post-hoc: sites closed/withdrawn
        else:
            status = "RECRUITING" if rng.random() < 0.6 else "ACTIVE_NOT_RECRUITING"
    elif rng.random() < 0.03 and start <= SNAPSHOT:
        status, why, enroll_n, enroll_type = "WITHDRAWN", "No participants enrolled", 0, "ACTUAL"
    elif end <= SNAPSHOT:
        if rng.random() < 0.05:
            status = "UNKNOWN"
        else:
            status = "COMPLETED"
            enroll_n, enroll_type = int(planned_n * rng.uniform(0.75, 1.15)), "ACTUAL"
            pc_date, pc_type = end, "ACTUAL"
    else:
        status = "RECRUITING" if rng.random() < 0.6 else "ACTIVE_NOT_RECRUITING"

    countries = list(rng.choice(COUNTRIES, size=min(len(COUNTRIES), max(1, int(np.log1p(n_sites)))), replace=False))
    locations = [{"facility": f"Site {k + 1}", "city": "City", "country": str(countries[k % len(countries)]),
                  **({"status": "RECRUITING"} if status == "RECRUITING" else {})} for k in range(n_sites)]
    drug = str(rng.choice(DRUGS))
    interventions = [{"type": "DRUG" if ph else str(rng.choice(["DEVICE", "BEHAVIORAL", "PROCEDURE"])), "name": drug if ph else "Intervention A"}]
    if rng.random() < 0.4:
        interventions.append({"type": "DRUG", "name": "Placebo"})
    last_update = min(SNAPSHOT, max(start, (pc_date if pc_type == "ACTUAL" else start) + timedelta(days=int(rng.integers(10, 400)))))
    refs = [{"pmid": str(20000000 + i), "type": "RESULT"}] if status == "COMPLETED" and rng.random() < 0.4 else []
    has_results = status in ("COMPLETED", "TERMINATED") and rng.random() < (0.55 if s_class == "INDUSTRY" else 0.25)

    branches = [{"abbrev": f"BC{area_i:02d}", "name": area}] + ([{"abbrev": "Rare", "name": "Rare Diseases"}] if rare else []) \
        + [{"abbrev": "All", "name": "All Conditions"}]
    status_mod = {"overallStatus": status, "startDateStruct": {"date": _fmt(start, rng), "type": "ACTUAL" if start <= SNAPSHOT else "ESTIMATED"},
                  "primaryCompletionDateStruct": {"date": _fmt(pc_date, rng), "type": pc_type},
                  "completionDateStruct": {"date": _fmt(pc_date + timedelta(days=90), rng), "type": pc_type},
                  "studyFirstSubmitDate": first_submit.isoformat(), "lastUpdatePostDateStruct": {"date": last_update.isoformat(), "type": "ACTUAL"}}
    if why:
        status_mod["whyStopped"] = why
    design = {"studyType": "INTERVENTIONAL", "phases": ph or ["NA"],
              "designInfo": {"allocation": str(rng.choice(["RANDOMIZED", "NON_RANDOMIZED", "NA"])),
                             "interventionModel": str(rng.choice(["PARALLEL", "SINGLE_GROUP", "CROSSOVER"])),
                             "primaryPurpose": str(rng.choice(["TREATMENT", "PREVENTION", "SUPPORTIVE_CARE"], p=[.75, .15, .1])),
                             "maskingInfo": {"masking": str(rng.choice(["NONE", "SINGLE", "DOUBLE", "QUADRUPLE"]))}},
              "enrollmentInfo": {"count": enroll_n, "type": enroll_type}}
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": f"A {' / '.join(ph) or 'Device'} Study of {drug} in {cond}",
                                     "organization": {"fullName": sponsor, "class": s_class}},
            "statusModule": status_mod,
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": sponsor, "class": s_class}},
            "conditionsModule": {"conditions": [cond]},
            "designModule": design,
            "armsInterventionsModule": {"armGroups": [{"label": f"Arm {k}"} for k in range(int(rng.integers(1, 4)))],
                                        "interventions": interventions},
            "eligibilityModule": {"eligibilityCriteria": elig, "healthyVolunteers": False, "sex": "ALL",
                                  "minimumAge": "18 Years", **({"maximumAge": "75 Years"} if rng.random() < 0.4 else {})},
            "contactsLocationsModule": {"locations": locations},
            "referencesModule": {"references": refs},
        },
        "derivedSection": {"conditionBrowseModule": {"meshes": [{"id": f"D{zlib.crc32(cond.encode()) % 999999:06d}", "term": cond}],
                                                     "browseBranches": branches},
                           "interventionBrowseModule": {"meshes": [{"id": "D000001", "term": drug}]} if ph else {}},
        "hasResults": bool(has_results),
    }


def write(studies: list[dict], name: str) -> str:
    path = config.LANDING_DIR / f"{name}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for s in studies:
            f.write(json.dumps(s) + "\n")
    return str(path)


def make_update(studies: list[dict], rng: np.random.Generator, update_day: date = date(2026, 10, 8)) -> list[dict]:
    """A later batch: some active trials finish/terminate, others slip their planned completion date."""
    active = [s for s in studies if s["protocolSection"]["statusModule"]["overallStatus"] in ("RECRUITING", "ACTIVE_NOT_RECRUITING")]
    out = []
    for s in rng.choice(active, size=min(300, len(active)), replace=False):
        s = json.loads(json.dumps(s))
        sm = s["protocolSection"]["statusModule"]
        u = rng.random()
        if u < 0.2:
            sm["overallStatus"], sm["whyStopped"] = "TERMINATED", str(rng.choice(REASONS["enrollment"]))
            sm["primaryCompletionDateStruct"] = {"date": update_day.isoformat(), "type": "ACTUAL"}
        elif u < 0.5:
            sm["overallStatus"] = "ACTIVE_NOT_RECRUITING"
        else:  # planned completion slips by 3-12 months
            d = sm["primaryCompletionDateStruct"]["date"]
            d = date.fromisoformat(d if len(d) == 10 else d + "-01")
            sm["primaryCompletionDateStruct"] = {"date": (d + timedelta(days=int(rng.integers(90, 365)))).isoformat(), "type": "ESTIMATED"}
        sm["lastUpdatePostDateStruct"] = {"date": update_day.isoformat(), "type": "ACTUAL"}
        out.append(s)
    return out


def generate(n: int, seed: int = config.SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [make_study(i, rng) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--update", action="store_true", help="write an incremental update batch instead")
    args = ap.parse_args()
    studies = generate(args.n)
    if args.update:
        print(write(make_update(studies, np.random.default_rng(config.SEED + 1)), "demo_incr_20261008"))
    else:
        print(write(studies, "demo_full_20261001"))


if __name__ == "__main__":
    main()
