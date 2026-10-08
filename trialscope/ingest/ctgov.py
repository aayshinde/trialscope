"""ClinicalTrials.gov API v2 client: full and incremental pulls into a gzipped JSONL landing zone.

The v2 API only serves the CURRENT version of each record (there is no version-history
endpoint), so every pull is kept as an immutable file and the warehouse snapshots
(dbt snapshot, SCD type 2) are what preserve how trials change over time.

    python -m trialscope.ingest.ctgov check                     # fetch 1 study, verify expected fields exist
    python -m trialscope.ingest.ctgov full [--limit 20000]      # first load (resumable: re-run to continue)
    python -m trialscope.ingest.ctgov incremental               # studies updated since the last load
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from datetime import date, datetime, timedelta

import requests

from trialscope import config

FIELDS = ("IdentificationModule,StatusModule,SponsorCollaboratorsModule,ConditionsModule,DesignModule,"
          "ArmsInterventionsModule,EligibilityModule,ContactsLocationsModule,ReferencesModule,DerivedSection,HasResults")
STATE = config.LANDING_DIR / "_state.json"
HEADERS = {"User-Agent": "TrialScope portfolio project (python-requests)"}

EXPECTED_PATHS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.startDateStruct.date",
    "protocolSection.statusModule.primaryCompletionDateStruct.type",
    "protocolSection.statusModule.lastUpdatePostDateStruct.date",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.enrollmentInfo.count",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "derivedSection.conditionBrowseModule.browseBranches",
    "hasResults",
]


def base_filter(since: str | None = None) -> str:
    f = f"AREA[StudyType]INTERVENTIONAL AND AREA[StartDate]RANGE[{config.MIN_START_DATE},MAX]"
    if since:
        f += f" AND AREA[LastUpdatePostDate]RANGE[{since},MAX]"
    return f


def get(session: requests.Session, url: str, params: dict, tries: int = 6) -> requests.Response:
    for i in range(tries):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code < 400 or r.status_code == 400:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            if i == tries - 1:
                raise
        time.sleep(min(60, 2 ** i) + random.random())
    raise RuntimeError(f"giving up on {url} after {tries} tries")


def pull(mode: str, since: str | None = None, limit: int | None = None, resume: bool = True) -> dict:
    """Page through /studies and append every study to a landing file. Resumable via _state.json."""
    session = requests.Session()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if resume and state.get("in_progress") and state.get("mode") == mode:
        out, token, n = config.LANDING_DIR / state["file"], state["next_token"], state["n"]
        since = state.get("since")
        print(f"Resuming {out.name} at {n:,} studies")
    else:
        out = config.LANDING_DIR / f"ctgov_{mode}_{datetime.now():%Y%m%dT%H%M%S}.jsonl.gz"
        token, n = None, 0

    params = {"format": "json", "pageSize": config.PAGE_SIZE, "fields": FIELDS, "filter.advanced": base_filter(since)}
    t0 = time.time()
    with gzip.open(out, "at", encoding="utf-8") as f:
        while True:
            p = dict(params, **({"pageToken": token} if token else {"countTotal": "true"}))
            r = get(session, f"{config.CTGOV_BASE}/studies", p)
            if r.status_code == 400 and "fields" in params:
                print(f"API rejected the field list ({r.text[:200]}); retrying with all fields")
                params.pop("fields")
                continue
            r.raise_for_status()
            js = r.json()
            if "totalCount" in js:
                print(f"{js['totalCount']:,} studies match ({mode}{', since ' + since if since else ''})")
            for s in js.get("studies", []):
                f.write(json.dumps(s) + "\n")
            n += len(js.get("studies", []))
            token = js.get("nextPageToken")
            STATE.write_text(json.dumps({"in_progress": bool(token), "mode": mode, "file": out.name, "next_token": token,
                                         "n": n, "since": since}))
            print(f"  {n:,} studies  ({n / max(time.time() - t0, 1):.0f}/s)", end="\r", flush=True)
            if not token or (limit and n >= limit):
                break
    STATE.write_text(json.dumps({"in_progress": False, "mode": mode, "file": out.name, "n": n, "since": since}))
    print(f"\nWrote {n:,} studies to {out}")
    return {"file": str(out), "n": n}


def last_watermark() -> str | None:
    """Latest lastUpdatePostDate already in the warehouse, minus an overlap window."""
    import duckdb

    if not config.WAREHOUSE.exists():
        return None
    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    try:
        d = con.execute("select max(last_update_date) from raw.studies").fetchone()[0]
    except duckdb.CatalogException:
        return None
    finally:
        con.close()
    return (d - timedelta(days=config.INCREMENTAL_OVERLAP_DAYS)).isoformat() if d else None


def check() -> None:
    r = get(requests.Session(), f"{config.CTGOV_BASE}/studies",
            {"format": "json", "pageSize": 1, "fields": FIELDS, "filter.advanced": base_filter()})
    print("HTTP", r.status_code)
    s = r.json()["studies"][0]
    for path in EXPECTED_PATHS:
        cur = s
        for k in path.split("."):
            cur = cur.get(k) if isinstance(cur, dict) else None
        print(f"  {'OK     ' if cur is not None else 'MISSING'} {path}" + (f" = {str(cur)[:60]}" if cur is not None else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "full", "incremental"])
    ap.add_argument("--limit", type=int, help="stop after about N studies (for a quicker first run)")
    ap.add_argument("--since", help="incremental: override watermark (YYYY-MM-DD)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.mode == "check":
        check()
    elif args.mode == "full":
        pull("full", limit=args.limit, resume=not args.no_resume)
    else:
        since = args.since or last_watermark() or (date.today() - timedelta(days=7)).isoformat()
        pull("incremental", since=since, limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
