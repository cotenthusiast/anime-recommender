"""Ranking metrics for top-*k* recommendation evaluation.

All functions follow the same convention:
- ``recs``:  ``{user_id: [item_id, …]}``        — predicted ranked lists.
- ``truth``: ``{user_id: set[item_id]}``         — ground-truth items per user.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(recs: dict[int, list[int]], truth: dict[int, set[int]], k: int) -> float:
    """Compute Recall@k averaged over all users.

    For each user, Recall@k = |hits| / |truth| where hits are the
    ground-truth items that appear in the top-k recommendations.

    Parameters
    ----------
    recs : dict[int, list[int]]
        Recommendations per user.
    truth : dict[int, set[int]]
        Ground-truth items per user.
    k : int
        Cut-off length.

    Returns
    -------
    float
        Recall@k averaged over all users in *truth*.
    """
    scores = []
    for user, actual in truth.items():
        recommendations = set(recs.get(user, [])[:k])
        hits = len(actual & recommendations)
        scores.append(hits / len(actual))
    return float(np.mean(scores))


def ndcg_at_k(recs: dict[int, list[int]], truth: dict[int, set[int]], k: int) -> float:
    """Compute NDCG@k averaged over all users.

    DCG sums ``1 / log2(rank + 1)`` for each hit. Ideal DCG assumes
    all ground-truth items appear at the top ranks.

    Parameters
    ----------
    recs : dict[int, list[int]]
        Recommendations per user.
    truth : dict[int, set[int]]
        Ground-truth items per user.
    k : int
        Cut-off length.

    Returns
    -------
    float
        NDCG@k averaged over all users in *truth*.
    """
    scores = []
    for user, actual in truth.items():
        recommendations = recs.get(user, [])[:k]
        dcg = 0.0
        for rank, item in enumerate(recommendations, start=1):
            if item in actual:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
        scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return float(np.mean(scores))