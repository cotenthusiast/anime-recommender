"""Content-based filtering recommender.

Recommends anime based on feature similarity rather than user behaviour.
Encodes each anime using its metadata (genres, studio, type, episodes)
and scores unseen items by cosine similarity to the user's watch history.
"""

from __future__ import annotations

import numpy as np

from anirec.models.base import Recommender


class ContentBased(Recommender):
    """Content-based recommender using anime metadata features.

    Builds a feature vector per anime from genre, studio, and type.
    A user profile is the mean feature vector of their rated items,
    weighted by rating. Scores candidates by cosine similarity to
    that profile.
    """

    def __init__(self, metadata_path: str) -> None:
        """Initialise with path to anime metadata.

        Parameters
        ----------
        metadata_path : str
            Path to Parquet file containing anime features
            (anime_id, genres, studio, type, episodes, etc.).
        """
        self.metadata_path = metadata_path
        self._item_features: np.ndarray | None = None  # (num_items, num_features)
        self._item_map: dict[int, int] = {}
        self._user_seen: dict[int, set[int]] = {}
        self._user_profiles: dict[int, np.ndarray] = {}

    def fit(self, train_path: str, **kwargs) -> None:
        """Build item feature matrix and user profiles from training ratings.

        Parameters
        ----------
        train_path : str
            Path to the training Parquet file.
        """
        ...

    def recommend(self, user_ids: list[int], k: int) -> dict[int, list[int]]:
        """Return top-k unseen items for each user by feature similarity.

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