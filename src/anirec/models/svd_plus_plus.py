"""SVD++ collaborative filtering recommender.

Extends SVD by incorporating implicit feedback: the set of items a
user has rated (regardless of the score) is itself a signal about
their preferences. Adds a per-item implicit factor y[i] that shifts
the user vector based on their rating history.
"""

from __future__ import annotations

import numpy as np

from anirec.models.base import Recommender


class SVDPlusPlus(Recommender):
    """Collaborative filter extending SVD with implicit feedback.

    Augments each user vector with a normalised sum of implicit item
    factors drawn from their full rating history, capturing the signal
    that rating something — regardless of score — reveals preference.
    """

    def __init__(
        self,
        k: int = 50,
        lr: float = 0.005,
        lambda_: float = 0.02,
        num_epochs: int = 10,
    ) -> None:
        """Initialise hyperparameters.

        Parameters
        ----------
        k : int
            Number of latent factors.
        lr : float
            Learning rate for SGD updates.
        lambda_ : float
            L2 regularisation coefficient.
        num_epochs : int
            Number of full passes over the training data.
        """
        self.k = k
        self.lr = lr
        self.lambda_ = lambda_
        self.num_epochs = num_epochs

        self.P: np.ndarray | None = None   # (num_users, k)
        self.Q: np.ndarray | None = None   # (num_items, k)
        self.Y: np.ndarray | None = None   # (num_items, k) implicit factors
        self._user_map: dict[int, int] = {}
        self._item_map: dict[int, int] = {}
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, train_path: str, **kwargs) -> None:
        """Learn P, Q, and implicit factor matrix Y from the training split.

        Parameters
        ----------
        train_path : str
            Path to the training Parquet file.
        """
        ...

    def recommend(self, user_ids: list[int], k: int) -> dict[int, list[int]]:
        """Return top-k unseen items for each user.

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