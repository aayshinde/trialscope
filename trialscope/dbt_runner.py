"""Run dbt programmatically (same on Windows/macOS/Linux, no shell needed).

    python -m trialscope.dbt_runner build        # snapshot + models + tests, in dependency order
    python -m trialscope.dbt_runner snapshot | run | test | docs
"""
from __future__ import annotations

import os
import sys

from trialscope import config


def dbt(*args: str) -> bool:
    from dbt.cli.main import dbtRunner

    os.environ["TRIALSCOPE_WAREHOUSE"] = str(config.WAREHOUSE.resolve())
    os.environ.setdefault("DBT_SEND_ANONYMOUS_USAGE_STATS", "False")
    cwd = os.getcwd()
    os.chdir(config.DBT_DIR)
    try:
        res = dbtRunner().invoke([*args, "--profiles-dir", str(config.DBT_DIR), "--project-dir", str(config.DBT_DIR)])
    finally:
        os.chdir(cwd)
    if res.exception:
        raise res.exception
    return bool(res.success)


def main() -> None:
    cmd = sys.argv[1:] or ["build"]
    if cmd == ["docs"]:
        cmd = ["docs", "generate"]
    ok = dbt(*cmd)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
