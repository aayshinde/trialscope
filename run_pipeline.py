"""Run the whole TrialScope pipeline in order (same on Windows, macOS, Linux).

    python run_pipeline.py --demo                 # synthetic data, ~2 minutes, no internet needed (to try it out)
    python run_pipeline.py --limit 30000          # real data, first 30K trials (~5-10 min download)
    python run_pipeline.py                        # real data, everything (~1-2 h download the first time)
    python run_pipeline.py --incremental          # daily update: new/changed trials only, then rebuild + rescore

The same steps are available as a scheduled Prefect flow: python flows/daily_flow.py
"""
import argparse
import subprocess
import sys
import time

STEPS_INGEST_FULL = [("Download trials (ClinicalTrials.gov API v2)", ["-m", "trialscope.ingest.ctgov", "full"])]
STEPS_INGEST_INCR = [("Download updated trials since last load", ["-m", "trialscope.ingest.ctgov", "incremental"])]
STEPS_DEMO = [("Generate synthetic demo trials", ["-m", "trialscope.demo_data"])]
STEPS_CORE = [
    ("Load landing files into DuckDB (raw)", ["-m", "trialscope.ingest.load"]),
    ("Sponsor entity resolution", ["-m", "trialscope.entity"]),
    ("dbt build: SCD2 snapshot, models, data tests", ["-m", "trialscope.dbt_runner", "build"]),
    ("Eligibility text features + embeddings", ["-m", "trialscope.text.eligibility"]),
    ("Termination reasons + topics", ["-m", "trialscope.text.termination"]),
]
STEPS_TRAIN = [
    ("Classifiers (out-of-time) + baselines + leakage ablation", ["-m", "trialscope.models.classifier"]),
    ("Survival analysis (KM, Cox, XGBoost-Cox)", ["-m", "trialscope.models.survival"]),
]
STEPS_SCORE = [
    ("Score trials + similar trials", ["-m", "trialscope.models.score"]),
    ("Drift monitoring (PSI)", ["-m", "trialscope.models.drift"]),
    ("Insights report", ["-m", "trialscope.insights"]),
    ("Weekly watchlist", ["-m", "trialscope.watchlist"]),
]


def run(steps):
    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n=== [{i}/{len(steps)}] {name} ===", flush=True)
        t0 = time.time()
        if subprocess.run([sys.executable, *cmd]).returncode != 0:
            sys.exit(f"Step failed: {name}")
        print(f"--- {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-download", action="store_true", help="reuse files already in data/landing")
    a = ap.parse_args()
    if a.demo:
        steps = STEPS_DEMO + STEPS_CORE + STEPS_TRAIN + STEPS_SCORE
    elif a.incremental:   # models are retrained on a schedule, not daily; rescore + drift every day
        steps = ([] if a.skip_download else STEPS_INGEST_INCR) + STEPS_CORE + STEPS_SCORE
    else:
        dl = [(n, c + (["--limit", str(a.limit)] if a.limit else [])) for n, c in STEPS_INGEST_FULL]
        steps = ([] if a.skip_download else dl) + STEPS_CORE + STEPS_TRAIN + STEPS_SCORE
    run(steps)
    print("\nDone. Open the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
