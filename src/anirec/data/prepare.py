"""Raw CSV → cleaned Parquet conversion."""

from __future__ import annotations

from pathlib import Path

import duckdb


def run(
    ratings_csv: str,
    user_col: str,
    item_col: str,
    rating_col: str,
    out_dir: str = "data/processed",
    items_csv: str | None = None,
    item_id_col: str | None = None,
    title_col: str | None = None,
    genres_col: str | None = None,
    sample_n: int = 200_000,
    threads: int = 4,
) -> None:
    """Convert raw CSVs into cleaned, standardised Parquet files.

    Parameters
    ----------
    ratings_csv : str
        Path to the ratings CSV file.
    user_col : str
        Column name for user IDs in the ratings CSV.
    item_col : str
        Column name for item IDs in the ratings CSV.
    rating_col : str
        Column name for ratings in the ratings CSV.
    out_dir : str
        Destination for Parquet outputs.
    items_csv : str | None
        Path to the anime metadata CSV. Skipped when None.
    item_id_col : str | None
        Column name for item IDs in the items CSV.
    title_col : str | None
        Column name for titles in the items CSV.
    genres_col : str | None
        Column name for genres in the items CSV. Optional.
    sample_n : int
        Number of rows to write to ``ratings_sample.parquet``.
    threads : int
        DuckDB thread count.
    """
    ratings_path = Path(ratings_csv)
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings CSV not found: {ratings_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ratings] using: {ratings_path}")
    print(f"[ratings] columns: user={user_col} item={item_col} rating={rating_col}")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)};")
    con.execute("PRAGMA enable_progress_bar=true;")

    con.execute("DROP TABLE IF EXISTS cleaned_ratings;")
    con.execute(
        f"""
        CREATE TABLE cleaned_ratings AS
        SELECT
            DENSE_RANK() OVER (ORDER BY u)  AS user_id,
            TRY_CAST(i AS BIGINT)           AS item_id,
            TRY_CAST(r AS DOUBLE)           AS rating
        FROM (
            SELECT
                CAST({user_col} AS VARCHAR)  AS u,
                {item_col}                   AS i,
                {rating_col}                 AS r
            FROM read_csv_auto('{ratings_path.as_posix()}', header=true)
        )
        WHERE u IS NOT NULL
          AND TRY_CAST(i AS BIGINT) IS NOT NULL
          AND TRY_CAST(r AS DOUBLE) IS NOT NULL
          AND r > 0
        """
    )

    stats = con.execute(
        """
        SELECT
            COUNT(*)                AS rows,
            COUNT(DISTINCT user_id) AS users,
            COUNT(DISTINCT item_id) AS items,
            MIN(rating)             AS rating_min,
            MAX(rating)             AS rating_max,
            AVG(rating)             AS rating_mean
        FROM cleaned_ratings
        """
    ).fetchone()
    print(
        f"[ratings] rows={stats[0]:,} users={stats[1]:,} items={stats[2]:,} "
        f"min={stats[3]} max={stats[4]} mean={float(stats[5]):.4f}"
    )

    ratings_out = out_dir / "ratings.parquet"
    con.execute(
        f"""
        COPY cleaned_ratings
        TO '{ratings_out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    print(f"[write] {ratings_out}")

    sample_out = out_dir / "ratings_sample.parquet"
    con.execute(
        f"""
        COPY (
            SELECT * FROM cleaned_ratings
            USING SAMPLE {int(sample_n)} ROWS
        )
        TO '{sample_out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    print(f"[write] {sample_out}")

    # ── items metadata (optional) ────────────────────────────────────
    if items_csv and item_id_col and title_col:
        items_path = Path(items_csv)
        if not items_path.exists():
            raise FileNotFoundError(f"Items CSV not found: {items_path}")

        print(f"[items] using: {items_path}")
        print(
            f"[items] columns: item_id={item_id_col} title={title_col}"
            + (f" genres={genres_col}" if genres_col else "")
        )

        genre_select = f", CAST({genres_col} AS VARCHAR) AS genres" if genres_col else ""
        con.execute("DROP TABLE IF EXISTS cleaned_items;")
        con.execute(
            f"""
            CREATE TABLE cleaned_items AS
            SELECT
                TRY_CAST({item_id_col} AS BIGINT) AS item_id,
                CAST({title_col} AS VARCHAR)      AS title
                {genre_select}
            FROM read_csv_auto('{items_path.as_posix()}', header=true)
            WHERE TRY_CAST({item_id_col} AS BIGINT) IS NOT NULL
              AND {title_col} IS NOT NULL
            """
        )

        items_out = out_dir / "items.parquet"
        con.execute(
            f"""
            COPY cleaned_items
            TO '{items_out.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )
        print(f"[write] {items_out}")

    con.close()
