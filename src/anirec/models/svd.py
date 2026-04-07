"""SVD-based collaborative filtering recommender.

Implements the classic Simon Funk SVD: learns user and item latent
factor matrices P and Q via SGD on observed ratings. Prediction for
a (user, item) pair is the dot product P[u] @ Q[i].
"""

from __future__ import annotations

import numpy as np
import pickle
import numba

from anirec.models.base import Recommender


@numba.njit()
def _sgd_loop(ratings, P, Q, b_u, b_i, mu, lr, lambda_):
    """Numba-compiled SGD inner loop over all ratings for one epoch.

    Parameters
    ----------
    ratings : np.ndarray
        Shape (n, 3) float64 array where column 0 is user index,
        column 1 is item index, and column 2 is the rating.
        Columns 0 and 1 must already be clean 0-based integer indices.
    P : np.ndarray
        User latent factor matrix, shape (num_users, k).
    Q : np.ndarray
        Item latent factor matrix, shape (num_items, k).
    b_u : np.ndarray
        User bias vector, shape (num_users,).
    b_i : np.ndarray
        Item bias vector, shape (num_items,).
    mu : float
        Global mean rating.
    lr : float
        Learning rate.
    lambda_ : float
        L2 regularisation coefficient.

    Returns
    -------
    tuple of (P, Q, b_u, b_i) with updated parameters.
    """
    for idx in range(len(ratings)):
        u = int(ratings[idx, 0])
        i = int(ratings[idx, 1])
        r = ratings[idx, 2]
        prediction = mu + b_i[i] + b_u[u] + np.dot(P[u], Q[i])
        error = r - prediction
        b_u[u] += lr * (error - lambda_ * b_u[u])
        b_i[i] += lr * (error - lambda_ * b_i[i])
        p_u = P[u].copy()
        P[u] += lr * (error * Q[i] - lambda_ * P[u])
        Q[i] += lr * (error * p_u - lambda_ * Q[i])
    return P, Q, b_u, b_i


class SVD(Recommender):
    """Collaborative filter via matrix factorisation with SGD.

    Learns k-dimensional latent vectors for every user and item such
    that their dot product approximates the observed rating.
    Includes user/item bias terms and a global mean for improved accuracy.
    """

    def __init__(
        self,
        k: int = 100,
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

        self.P: np.ndarray | None = None       # (num_users, k)
        self.Q: np.ndarray | None = None       # (num_items, k)
        self.b_u: np.ndarray | None = None     # (num_users,)
        self.b_i: np.ndarray | None = None     # (num_items,)
        self.mu: float = 0.0                   # global mean
        self._user_map: dict[int, int] = {}
        self._item_map: dict[int, int] = {}
        self._item_map_reverse: dict[int, int] = {}
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, train_path: str, **kwargs) -> None:
        """Learn P, Q, and bias terms from the training split.

        Parameters
        ----------
        train_path : str
            Path to the training Parquet file.
        """
        ratings = self._load_ratings(train_path)
        self._build_maps(ratings)
        P, Q, b_u, b_i, mu = self._init_matrices(
            len(self._user_map), len(self._item_map), ratings
        )
        self.P, self.Q, self.b_u, self.b_i, self.mu = self._train(
            ratings, P, Q, b_u, b_i, mu
        )

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
        results = {}
        for uid in user_ids:
            if uid not in self._user_map:
                results[uid] = []
                continue
            u = self._user_map[uid]
            predictions = self.mu + self.b_u[u] + self.b_i + self.P[u] @ self.Q.T
            predictions[list(self._user_seen[u])] = -np.inf
            top_k_indices = np.argsort(predictions)[-k:][::-1]
            results[uid] = [self._item_map_reverse[i] for i in top_k_indices]
        return results

    def save(self, path: str) -> None:
        """Save model matrices and maps to disk.

        Parameters
        ----------
        path : str
            Base path — saves {path}.npz and {path}.pkl.
        """
        np.savez(f"{path}.npz", P=self.P, Q=self.Q, b_u=self.b_u, b_i=self.b_i)
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "user_map": self._user_map,
                "item_map": self._item_map,
                "item_map_reverse": self._item_map_reverse,
                "user_seen": self._user_seen,
                "mu": self.mu,
            }, f)

    def load(self, path: str) -> None:
        """Load model matrices and maps from disk.

        Parameters
        ----------
        path : str
            Base path — loads {path}.npz and {path}.pkl.
        """
        data = np.load(f"{path}.npz")
        self.P = data["P"]
        self.Q = data["Q"]
        self.b_u = data["b_u"]
        self.b_i = data["b_i"]
        with open(f"{path}.pkl", "rb") as f:
            maps = pickle.load(f)
        self._user_map = maps["user_map"]
        self._item_map = maps["item_map"]
        self._item_map_reverse = maps["item_map_reverse"]
        self._user_seen = maps["user_seen"]
        self.mu = maps["mu"]

    def _load_ratings(self, train_path: str) -> np.ndarray:
        """Load ratings from Parquet, return as np array of shape (n, 3)."""
        import duckdb
        con = duckdb.connect()
        rows = con.execute(f"""
            SELECT user_id, item_id, rating
            FROM read_parquet('{train_path}')
        """).fetchall()
        con.close()
        return np.array(rows, dtype=np.float32)

    def _build_maps(self, ratings: np.ndarray) -> None:
        """Build _user_map, _item_map, _item_map_reverse, and _user_seen."""
        user_ids = list(set(int(x) for x in ratings[:, 0]))
        item_ids = list(set(int(x) for x in ratings[:, 1]))
        self._user_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self._item_map = {iid: idx for idx, iid in enumerate(item_ids)}
        self._item_map_reverse = {idx: iid for iid, idx in self._item_map.items()}
        for row in ratings:
            u = self._user_map[int(row[0])]
            i = self._item_map[int(row[1])]
            if u not in self._user_seen:
                self._user_seen[u] = set()
            self._user_seen[u].add(i)

    def _init_matrices(
        self, num_users: int, num_items: int, ratings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Initialise P, Q, bias vectors, and global mean.

        Parameters
        ----------
        num_users : int
            Number of unique users in training.
        num_items : int
            Number of unique items in training.
        ratings : np.ndarray
            Full training ratings array, used to compute global mean.
        """
        P = np.random.normal(0, 0.01, (num_users, self.k))
        Q = np.random.normal(0, 0.01, (num_items, self.k))
        b_u = np.zeros(num_users, dtype=np.float64)
        b_i = np.zeros(num_items, dtype=np.float64)
        mu = float(ratings[:, 2].mean())
        return P, Q, b_u, b_i, mu

    def _train(
        self,
        ratings: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        b_u: np.ndarray,
        b_i: np.ndarray,
        mu: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Pre-convert ratings to integer indices, then run SGD for all epochs.

        Builds a float64 array with user and item columns mapped to clean
        0-based indices via _user_map and _item_map. Each epoch shuffles
        a copy of this array and delegates to the Numba-compiled _sgd_loop.

        Returns fitted (P, Q, b_u, b_i, mu).
        """
        ratings_idx = np.array(
            [[self._user_map[int(row[0])], self._item_map[int(row[1])], row[2]] for row in ratings],
            dtype=np.float64
        )       
        for epoch in range(self.num_epochs):
            temp_ratings = ratings_idx.copy()
            np.random.shuffle(temp_ratings)
            print(f"[train] epoch {epoch + 1}/{self.num_epochs}")
            P, Q, b_u, b_i = _sgd_loop(temp_ratings, P, Q, b_u, b_i, mu, self.lr, self.lambda_)
        return P, Q, b_u, b_i, mu
    

