"""Shared helpers for loading Parquet splits into DuckDB or Python.

Every model and evaluation script should go through these functions
rather than hardcoding ``read_parquet(...)`` calls.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def load_parquet(path: str | Path, threads: int = 4) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with the Parquet file registered as a ``data`` view.

    Parameters
    ----------
    path : str | Path
        Path to a ``.parquet`` file.
    threads : int
        DuckDB thread count.

    Returns
    -------
    duckdb.DuckDBPyConnection
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"""
        CREATE VIEW data AS
        SELECT * FROM read_parquet('{path.as_posix()}')
    """)
    return con


def load_truth(test_path: str | Path) -> dict[int, set[int]]:
    """Load a test/val split into a ``{user_id: set[item_id]}`` lookup.

    Supports multiple ground-truth items per user.

    Parameters
    ----------
    test_path : str | Path
        Path to the test or validation Parquet file.

    Returns
    -------
    dict[int, set[int]]
        Mapping from user ID to set of ground-truth item IDs.
    """
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT user_id, item_id
        FROM read_parquet('{Path(test_path).as_posix()}')
    """).fetchall()
    con.close()
    truth: dict[int, set[int]] = {}
    for uid, iid in rows:
        uid, iid = int(uid), int(iid)
        if uid not in truth:
            truth[uid] = set()
        truth[uid].add(iid)
    return truth