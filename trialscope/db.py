"""Small DuckDB helpers shared by the Python steps (the dbt models own raw -> marts)."""
from __future__ import annotations

import duckdb
import pandas as pd

from trialscope import config


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """DuckDB allows one configuration per file per process. If something in this process (e.g. dbt) already
    holds a read-write connection, a read-only request reuses read-write mode instead of failing."""
    try:
        return duckdb.connect(str(config.WAREHOUSE), read_only=read_only)
    except duckdb.ConnectionException:
        if read_only:
            return duckdb.connect(str(config.WAREHOUSE), read_only=False)
        raise


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def write_table(df: pd.DataFrame, table: str, schema: str = "ml") -> None:
    con = connect()
    try:
        con.execute(f"create schema if not exists {schema}")
        con.register("_df", df)
        con.execute(f"create or replace table {schema}.{table} as select * from _df")
        con.unregister("_df")
    finally:
        con.close()


def table_exists(table: str, schema: str = "ml") -> bool:
    try:
        return query("select count(*) n from information_schema.tables where table_schema=? and table_name=?",
                     [schema, table]).n[0] > 0
    except duckdb.IOException:
        return False
