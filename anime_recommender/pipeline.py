from pathlib import Path
import pandas as pd
from .dataio import load_ratings

def preprocess(ratings: pd.DataFrame) -> pd.DataFrame:
    df = ratings.copy()
    df = df.dropna(subset=["user_id", "anime_id", "rating"])
    df["user_id"] = df["user_id"].astype(int)
    df["anime_id"] = df["anime_id"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].clip(lower=0, upper=10)
    return df

def recommend_top_n_by_mean(ratings: pd.DataFrame, n: int = 5, min_count: int = 2) -> pd.DataFrame:
    agg = (
        ratings.groupby("anime_id")["rating"]
        .agg(mean_rating="mean", ratings_count="count")
        .reset_index()
    )
    agg = agg[agg["ratings_count"] >= min_count]
    agg = agg.sort_values(["mean_rating", "ratings_count"], ascending=[False, False]).head(n)
    return agg

def run(sample_csv: str | Path | None = None) -> None:
    root = Path(__file__).resolve().parent.parent
    csv_path = Path(sample_csv) if sample_csv else (root / "data" / "sample_ratings.csv")

    ratings = load_ratings(csv_path)
    ratings = preprocess(ratings)

    top = recommend_top_n_by_mean(ratings, n=5, min_count=2)
    print("Top-N baseline (by mean rating):")
    print(top.to_string(index=False))
