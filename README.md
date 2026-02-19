# Anime Recommender System

**Status: On Hold (Archived)**

This project is temporarily paused while I focus on building core machine learning foundations and smaller experimental projects. The repository remains available as a reference for the intended architecture and future development.

Planned future work includes:
- Dataset integration and preprocessing pipeline
- Baseline recommendation models
- Evaluation framework (ranking metrics)
- Experiment tracking and reproducibility tooling

Development is expected to resume in a future iteration.


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
