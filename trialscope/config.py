"""Central configuration. Every value can be overridden with an environment variable or .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _p(name: str, default: Path) -> Path:
    return Path(os.getenv(name, default))


DATA_DIR = _p("DATA_DIR", ROOT / "data")
LANDING_DIR = DATA_DIR / "landing"            # raw API pages, gzipped JSONL, one file per run
WAREHOUSE = _p("WAREHOUSE", ROOT / "warehouse" / "trialscope.duckdb")
MODEL_DIR = _p("MODEL_DIR", ROOT / "models")
RESULTS_DIR = _p("RESULTS_DIR", ROOT / "results")
LABELS_DIR = _p("LABELS_DIR", ROOT / "labels")
DBT_DIR = ROOT / "dbt"

# ---- ingestion -----------------------------------------------------------------
CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
MIN_START_DATE = os.getenv("MIN_START_DATE", "2000-01-01")   # ingest from 2000 so 5-year sponsor history is full by 2008
MODEL_MIN_START_YEAR = int(os.getenv("MODEL_MIN_START_YEAR", "2008"))  # modelling cohorts start here
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "1000"))
INCREMENTAL_OVERLAP_DAYS = int(os.getenv("INCREMENTAL_OVERLAP_DAYS", "3"))
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")                  # optional: raises PubMed rate limit 3 -> 10 req/s
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")            # optional

# ---- modelling -----------------------------------------------------------------
TRAIN_END_YEAR = int(os.getenv("TRAIN_END_YEAR", "2015"))     # train: start year <= 2015
VAL_YEAR = int(os.getenv("VAL_YEAR", "2016"))                 # validation (threshold/early stopping)
TEST_START_YEAR = int(os.getenv("TEST_START_YEAR", "2017"))   # out-of-time test cohort
TEST_END_YEAR = int(os.getenv("TEST_END_YEAR", "2019"))       # recent cohorts are mostly unresolved
WITHDRAWN_AS_FAILURE = os.getenv("WITHDRAWN_AS_FAILURE", "0") == "1"
SEED = int(os.getenv("SEED", "42"))

# ---- LLM / embeddings ----------------------------------------------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "local")     # "local" = TF-IDF + SVD; or text-embedding-3-small
EMBED_DIM = int(os.getenv("EMBED_DIM", "64"))
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-3.5-turbo": (0.50, 1.50)}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

for d in (DATA_DIR, LANDING_DIR, WAREHOUSE.parent, MODEL_DIR, RESULTS_DIR, LABELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
