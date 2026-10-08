"""Flatten one ClinicalTrials.gov v2 study JSON into relational rows. Defensive: any field may be missing."""
from __future__ import annotations

import hashlib
import re
from datetime import date


# The live API no longer returns conditionBrowseModule.browseBranches, only MeSH meshes + ancestors.
# Map MeSH tree terms to the ClinicalTrials.gov branch names the dbt models expect.
MESH_TO_BRANCH = {
    "Neoplasms": "Neoplasms",
    "Cardiovascular Diseases": "Heart and Blood Diseases", "Hematologic Diseases": "Heart and Blood Diseases",
    "Nervous System Diseases": "Nervous System Diseases",
    "Mental Disorders": "Behaviors and Mental Disorders", "Substance-Related Disorders": "Behaviors and Mental Disorders",
    "Infections": "Infections", "Virus Diseases": "Infections", "Bacterial Infections and Mycoses": "Infections",
    "Parasitic Diseases": "Infections",
    "Nutritional and Metabolic Diseases": "Nutritional and Metabolic Diseases",
    "Immune System Diseases": "Immune System Diseases",
    "Respiratory Tract Diseases": "Respiratory Tract (Lung and Bronchial) Diseases",
    "Digestive System Diseases": "Digestive System Diseases",
    "Skin and Connective Tissue Diseases": "Skin and Connective Tissue Diseases",
    "Musculoskeletal Diseases": "Muscle, Bone, and Cartilage Diseases",
    "Urogenital Diseases": "Urinary Tract, Sexual Organs, and Pregnancy Conditions",
    "Pregnancy Complications": "Urinary Tract, Sexual Organs, and Pregnancy Conditions",
    "Endocrine System Diseases": "Gland and Hormone Related Diseases",
    "Eye Diseases": "Eye Diseases", "Otorhinolaryngologic Diseases": "Ear, Nose, and Throat Diseases",
    "Stomatognathic Diseases": "Mouth and Tooth Diseases", "Wounds and Injuries": "Wounds and Injuries",
    "Congenital, Hereditary, and Neonatal Diseases and Abnormalities":
        "Congenital, Hereditary, and Neonatal Diseases and Abnormalities",
    "Chemically-Induced Disorders": "Chemically-Induced Disorders",
    "Signs and Symptoms": "Symptoms and General Pathology", "Pathological Conditions, Anatomical": "Symptoms and General Pathology",
}


# Trials with no MeSH terms at all (about 1 in 5) are classified from their free-text condition names.
CONDITION_KEYWORDS = [
    (r"cancer|carcinoma|tumou?r|neoplasm|lymphoma|leuk(a)?emia|melanoma|sarcoma|myeloma|glioma|metasta|malignan|oncolog",
     "Neoplasms"),
    (r"heart|cardi|hypertens|coronary|atrial|anemia|anaemia|thromb|hemophilia|sickle|vascular|hemorrhage", "Heart and Blood Diseases"),
    (r"alzheimer|parkinson|epilep|migraine|neuropath|multiple sclerosis|dementia|stroke|seizure|spinal cord", "Nervous System Diseases"),
    (r"depress|anxiety|schizophren|bipolar|adhd|autism|ptsd|insomnia|alcohol|opioid|smoking|addiction|mental",
     "Behaviors and Mental Disorders"),
    (r"infection|hiv|covid|sars-cov|hepatitis|influenza|tubercul|malaria|sepsis|pneumonia|vaccin|virus|bacteri", "Infections"),
    (r"diabet|obes|metabolic|lipid|cholesterol|overweight|insulin", "Nutritional and Metabolic Diseases"),
    (r"arthritis|lupus|allerg|crohn|colitis|immun|transplant|graft", "Immune System Diseases"),
    (r"asthma|copd|pulmonary|lung disease|cystic fibrosis|bronch|respiratory", "Respiratory Tract (Lung and Bronchial) Diseases"),
    (r"liver|hepat|gastr|bowel|pancrea|nafld|nash|reflux|cirrhosis", "Digestive System Diseases"),
    (r"dermat|eczema|acne|psoria|skin|scleroderma", "Skin and Connective Tissue Diseases"),
    (r"osteo|fracture|arthroplasty|muscle|bone|sarcopenia|tendon|back pain|joint|knee|hip", "Muscle, Bone, and Cartilage Diseases"),
    (r"kidney|renal|prostat|bladder|urinary|pregnan|preeclampsia|endometri|infertil|contracepti|menopaus|cesarean",
     "Urinary Tract, Sexual Organs, and Pregnancy Conditions"),
    (r"\beye\b|glaucoma|macular|retin|cataract|myopia|ocular", "Eye Diseases"),
    (r"healthy (volunteer|subject|participant|adult|men|women)", "Healthy Volunteers"),
    (r"pain|fever|nausea|fatigue|pruritus", "Symptoms and General Pathology"),
]
# ClinicalTrials.gov's own "Rare Diseases" branch (NIH GARD list) is no longer exposed. Inherited / inborn-error MeSH
# categories are used as a PROXY for it, so `is_rare_disease` means "genetic or inborn disease", not the official flag.
RARE_PROXY_TERMS = {"Genetic Diseases, Inborn", "Metabolism, Inborn Errors"}


def branches_from_mesh(cb: dict, conditions: list[str] | None = None) -> list[dict]:
    """Browse branches as the API returns them, else derived from MeSH meshes + ancestors, else from condition keywords."""
    if cb.get("browseBranches"):
        return [{"abbrev": b.get("abbrev"), "name": b.get("name")} for b in cb["browseBranches"]]
    terms = {m["term"] for kind in ("meshes", "ancestors") for m in cb.get(kind, []) if m.get("term")}
    names = {MESH_TO_BRANCH[t] for t in terms if t in MESH_TO_BRANCH}
    if not names:
        text = " ".join(conditions or []).lower()
        names = {branch for pattern, branch in CONDITION_KEYWORDS if re.search(pattern, text)}
    if terms & RARE_PROXY_TERMS:
        names.add("Rare Diseases")
    return [{"abbrev": None, "name": n} for n in sorted(names)]


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def parse_date(s: str | None) -> date | None:
    """'2019', '2019-03' or '2019-03-15' -> date (month/year precision mapped to the first day)."""
    if not s:
        return None
    parts = s.split("-")
    try:
        return date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1, int(parts[2]) if len(parts) > 2 else 1)
    except (ValueError, IndexError):
        return None


AGE_RE = re.compile(r"([\d.]+)\s*(year|month|week|day|hour|minute)", re.I)
AGE_DIV = {"year": 1, "month": 12, "week": 52.18, "day": 365.25, "hour": 8766, "minute": 525960}


def parse_age(s: str | None) -> float | None:
    m = AGE_RE.search(s or "")
    return round(float(m.group(1)) / AGE_DIV[m.group(2).lower()], 3) if m else None


def parse_study(js: dict) -> dict:
    ps = js.get("protocolSection", {})
    nct = g(ps, "identificationModule", "nctId")
    sm, dm, em = g(ps, "statusModule", default={}), g(ps, "designModule", default={}), g(ps, "eligibilityModule", default={})
    lead = g(ps, "sponsorCollaboratorsModule", "leadSponsor", default={})
    locs = g(ps, "contactsLocationsModule", "locations", default=[])
    phases = [p for p in g(dm, "phases", default=[]) if p != "NA"]
    countries = sorted({l.get("country") for l in locs if l.get("country")})
    study = {
        "nct_id": nct,
        "brief_title": g(ps, "identificationModule", "briefTitle"),
        "overall_status": sm.get("overallStatus"),
        "why_stopped": sm.get("whyStopped"),
        "start_date": parse_date(g(sm, "startDateStruct", "date")),
        "start_date_type": g(sm, "startDateStruct", "type"),
        "primary_completion_date": parse_date(g(sm, "primaryCompletionDateStruct", "date")),
        "primary_completion_type": g(sm, "primaryCompletionDateStruct", "type"),
        "completion_date": parse_date(g(sm, "completionDateStruct", "date")),
        "completion_type": g(sm, "completionDateStruct", "type"),
        "first_submit_date": parse_date(sm.get("studyFirstSubmitDate")),
        "last_update_date": parse_date(g(sm, "lastUpdatePostDateStruct", "date")),
        "lead_sponsor_name": lead.get("name"),
        "lead_sponsor_class": lead.get("class"),
        "n_collaborators": len(g(ps, "sponsorCollaboratorsModule", "collaborators", default=[])),
        "study_type": dm.get("studyType"),
        "phase": "|".join(phases) if phases else "NA",
        "allocation": g(dm, "designInfo", "allocation"),
        "intervention_model": g(dm, "designInfo", "interventionModel"),
        "primary_purpose": g(dm, "designInfo", "primaryPurpose"),
        "masking": g(dm, "designInfo", "maskingInfo", "masking"),
        "enrollment_count": g(dm, "enrollmentInfo", "count"),
        "enrollment_type": g(dm, "enrollmentInfo", "type"),
        "n_arms": len(g(ps, "armsInterventionsModule", "armGroups", default=[])),
        "eligibility_criteria": em.get("eligibilityCriteria"),
        "healthy_volunteers": em.get("healthyVolunteers"),
        "sex": em.get("sex"),
        "min_age_years": parse_age(em.get("minimumAge")),
        "max_age_years": parse_age(em.get("maximumAge")),
        "n_locations": len(locs),
        "n_countries": len(countries),
        "countries": "|".join(countries),
        "has_results": bool(js.get("hasResults", False)),
    }
    tracked = {k: study[k] for k in ("overall_status", "why_stopped", "start_date", "primary_completion_date",
                                     "primary_completion_type", "enrollment_count", "enrollment_type", "phase",
                                     "lead_sponsor_name", "n_locations")}
    study["record_hash"] = hashlib.md5(repr(sorted(tracked.items())).encode()).hexdigest()

    cb, ib = g(js, "derivedSection", "conditionBrowseModule", default={}), g(js, "derivedSection", "interventionBrowseModule", default={})
    return {
        "studies": [study],
        "conditions": [{"nct_id": nct, "condition": c} for c in g(ps, "conditionsModule", "conditions", default=[])],
        "condition_mesh": [{"nct_id": nct, "mesh_id": m.get("id"), "term": m.get("term"), "kind": kind}
                           for kind in ("meshes", "ancestors") for m in cb.get(kind, [])],
        "browse_branches": [{"nct_id": nct, **b} for b in branches_from_mesh(cb, g(ps, "conditionsModule", "conditions", default=[]))],
        "interventions": [{"nct_id": nct, "type": i.get("type"), "name": i.get("name")}
                          for i in g(ps, "armsInterventionsModule", "interventions", default=[])],
        "intervention_mesh": [{"nct_id": nct, "mesh_id": m.get("id"), "term": m.get("term")} for m in ib.get("meshes", [])],
        "sites": [{"nct_id": nct, "facility": l.get("facility"), "city": l.get("city"), "country": l.get("country"),
                   "site_status": l.get("status")} for l in locs],
        "references": [{"nct_id": nct, "pmid": r.get("pmid"), "ref_type": r.get("type")}
                       for r in g(ps, "referencesModule", "references", default=[])],
    }
