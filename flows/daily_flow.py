"""Prefect orchestration.

    python flows/daily_flow.py --once            # run the daily flow now
    python flows/daily_flow.py --serve           # schedule: daily update 06:00, weekly retrain Sunday 03:00

daily:   incremental download -> load -> entity resolution -> dbt build (SCD2 snapshot + tests)
         -> text features -> rescore -> drift check
weekly:  daily steps + retrain classifier and survival model
"""
import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parent.parent


@task(retries=2, retry_delay_seconds=120)
def step(*args: str) -> None:
    log = get_run_logger()
    log.info("running %s", " ".join(args))
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)
    log.info(r.stdout[-3000:])
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])


@flow(name="trialscope-daily", log_prints=True)
def daily(download: bool = True) -> None:
    if download:
        step("trialscope.ingest.ctgov", "incremental")
    step("trialscope.ingest.load")
    step("trialscope.entity")
    step("trialscope.dbt_runner", "build")       # fails the flow if a data test fails
    step("trialscope.text.eligibility")
    step("trialscope.text.termination")
    step("trialscope.models.score")
    step("trialscope.models.drift")
    step("trialscope.watchlist")                 # what changed in the top-risk list since the last run


@flow(name="trialscope-weekly-retrain", log_prints=True)
def weekly(download: bool = True) -> None:
    daily(download)
    step("trialscope.models.classifier")
    step("trialscope.models.survival")
    step("trialscope.models.score")
    step("trialscope.insights")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    if a.serve:
        from prefect import serve
        serve(daily.to_deployment(name="daily", cron="0 6 * * *"),
              weekly.to_deployment(name="weekly", cron="0 3 * * 0"))
    else:
        daily(download=not a.no_download)
