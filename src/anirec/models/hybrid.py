"""Hybrid recommender combining collaborative and content-based signals.

Blends scores from a trained SVD (or SVD++) model and a ContentBased
model via a weighted sum. The best-performing individual models are
selected after evaluation and combined here.
"""

from __future__ import annotations

from anirec.models.base import Recommender
from anirec.models.svd import SVD
from anirec.models.content_based import ContentBased


class Hybrid(Recommender):
    """Weighted hybrid of collaborative filtering and content-based scoring.

    Combines normalised scores from an SVD model and a ContentBased
    model: final_score = alpha * svd_score + (1 - alpha) * cb_score.
    Alpha is tuned on the validation set.
    """

    def __init__(
        self,
        svd_model: SVD,
        cb_model: ContentBased,
        alpha: float = 0.7,
    ) -> None:
        """Initialise with pre-trained component models.

        Parameters
        ----------
        svd_model : SVD
            A fitted SVD or SVDPlusPlus instance.
        cb_model : ContentBased
            A fitted ContentBased instance.
        alpha : float
            Weight given to the collaborative score (0-1).
            1 - alpha is given to the content-based score.
        """
        self.svd_model = svd_model
        self.cb_model = cb_model
        self.alpha = alpha

    def fit(self, train_path: str) -> None:
        """Fit both component models on the training split.

        Parameters
        ----------
        train_path : str
            Path to the training Parquet file.
        """
        ...

    def recommend(self, user_ids: list[int], k: int) -> dict[int, list[int]]:
        """Return top-k items by blended score for each user.

        Parameters
        ----------
        user_ids : list[int]
            Original user IDs to generate recommendations for.
        k : int
            Number of items to recommend per user.

        Returns
        -------
        dict[int, list[int]]
            Mapping of user_id to list of recommended item_ids.
        """
        ...