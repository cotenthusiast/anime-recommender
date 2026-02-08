from pathlib import Path
import pandas as pd

def load_ratings(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"user_id","anime_id","rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df