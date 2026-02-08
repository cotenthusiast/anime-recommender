# Anime Recommender (WIP)

Status: Work in progress.

## What runs now
- Loads ratings from CSV (ingestion)
- Basic cleaning/typing (preprocessing)
- Baseline recommender: top-N by mean rating
- Prints results to console

## Run
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m anime_recommender
```
## Planned
- Proper train/validation split
- Offline evaluation: Precision@K, Recall@K
- Better baselines + content/collab methods