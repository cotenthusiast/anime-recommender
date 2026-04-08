#!/usr/bin/env python3
"""CLI wrapper for data preparation (raw CSV → Parquet).

Usage
-----
    python scripts/prepare.py
    python scripts/prepare.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from anirec.config import load_config
from anirec.data.prepare import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert raw CSVs to cleaned Parquet.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["paths"]
    prep = cfg["prepare"]

    run(
        ratings_csv=prep["ratings_csv"],
        user_col=prep["user_col"],
        item_col=prep["item_col"],
        rating_col=prep["rating_col"],
        out_dir=p["processed_dir"],
        items_csv=prep.get("items_csv"),
        item_id_col=prep.get("item_id_col"),
        title_col=prep.get("title_col"),
        genres_col=prep.get("genres_col"),
        sample_n=prep["sample_n"],
        threads=prep["threads"],
    )


if __name__ == "__main__":
    main()
