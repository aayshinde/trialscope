"""Point every path at a temp dir BEFORE trialscope.config is imported."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="trialscope_test_"))
os.environ.update({
    "DATA_DIR": str(_tmp / "data"), "WAREHOUSE": str(_tmp / "wh" / "test.duckdb"), "MODEL_DIR": str(_tmp / "models"),
    "RESULTS_DIR": str(_tmp / "results"), "MLFLOW_TRACKING_URI": f"sqlite:///{_tmp / 'mlflow.db'}",
    "EMBED_MODEL": "local", "EMBED_DIM": "16", "MLFLOW_DISABLE_AGENT_HINT": "1", "DBT_SEND_ANONYMOUS_USAGE_STATS": "False",
})
# empty (not unset) so load_dotenv() cannot pull a developer's real .env values into the tests
os.environ.update({"OPENAI_API_KEY": "", "OPENAI_BASE_URL": "", "CHAT_MODEL": "gpt-4o-mini"})
