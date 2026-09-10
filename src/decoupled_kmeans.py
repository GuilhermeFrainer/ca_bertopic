"""
Decoupled Multi-View K-Means Clustering.

Allows specifying independent distance metrics (e.g., Cosine for textual embeddings
and Euclidean for tabular metadata) and modality weights for each view.
"""

import warnings
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from mvlearn.cluster.base import BaseCluster
from mvlearn.utils.utils import check_Xs
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.preprocessing import normalize


class DecoupledMultiviewKMeans(BaseCluster):
    """
    Multi-View K-Means with decoupled per-view distance metrics.

    Enables directional metrics (e.g. Cosine distance with spherical centroid updates)
    on semantic representations (View 0) while applying metric-space distances
    (e.g. Euclidean distance with standard centroid means) on tabular metadata (View 1).

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    random_state : int, optional, default=None
        Determines random number generation for centroid initialization.
    init : {'k-means++', 'random'}, default='k-means++'
        Centroid initialization method.
    patience : int, default=5
        Number of iterations with improvement < tol before early stopping.
    max_iter : int, optional, default=300
        Maximum number of iterations of expectation-maximization.
    n_init : int, default=5
        Number of times the algorithm will run with different centroid seeds.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    n_jobs : int, optional, default=None
        Number of jobs to run in parallel. None means 1.
    view_metrics : Sequence[str], default=('cosine', 'euclidean')
        Distance metric to use for each view. Supported: 'cosine', 'euclidean'.
    view_weights : Sequence[float], optional, default=(1.0, 1.0)
        Relative importance weighting assigned to each view during assignment.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        random_state: Optional[int] = None,
        init: str = "k-means++",
        patience: int = 5,
        max_iter: Optional[int] = 300,
        n_init: int = 5,
        tol: float = 1e-4,
        n_jobs: Optional[int] = None,
        view_metrics: Sequence[str] = ("cosine", "euclidean"),
        view_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.init = init
        self.patience = patience
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.n_jobs = n_jobs
        if isinstance(view_metrics, str):
            self.view_metrics = [m.strip() for m in view_metrics.split(",")]
        else:
            self.view_metrics = list(view_metrics)

        if isinstance(view_weights, str):
            self.view_weights = [float(w.strip()) for w in view_weights.split(",")]
        elif view_weights is not None:
            self.view_weights = [float(w) for w in view_weights]
        else:
            self.view_weights = [1.0] * len(self.view_metrics)

        self.centroids_: Optional[List[np.ndarray]] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def _preprocess_data(self, Xs: Sequence[Any]) -> List[np.ndarray]:
        """Validate input views."""
        Xs_checked = check_Xs(Xs, enforce_views=2)
        if len(self.view_metrics) != 2:
            raise ValueError(
                f"view_metrics must have length 2 matching views, "
                f"got {len(self.view_metrics)}"
            )
        if len(self.view_weights) != 2:
            raise ValueError(
                f"view_weights must have length 2 matching views, "
                f"got {len(self.view_weights)}"
            )
        return [np.asarray(x, dtype=np.float64) for x in Xs_checked]

    def _compute_view_dist(
        self, X: np.ndarray, centers: np.ndarray, view_idx: int
    ) -> np.ndarray:
        """Computes pairwise distance between samples X and centers for a view."""
        metric = self.view_metrics[view_idx].lower()
        if metric == "cosine":
            X_norm = normalize(X)
            C_norm = normalize(centers)
            sim = X_norm @ C_norm.T
            return np.clip(1.0 - sim, 0.0, 2.0)
        elif metric == "euclidean":
            return cdist(X, centers, metric="euclidean")
        else:
            return cdist(X, centers, metric=metric)

    def _init_centroids(
        self, Xs: List[np.ndarray], rng: np.random.RandomState
    ) -> List[np.ndarray]:
        """Initializes centroids for both views."""
        n_samples = Xs[0].shape[0]
        if self.init == "random":
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            c0 = Xs[0][indices].copy()
            c1 = Xs[1][indices].copy()
        elif self.init == "k-means++":
            indices = [rng.randint(n_samples)]
            c1_list = [Xs[1][indices[0]]]

            for _ in range(self.n_clusters - 1):
                cur_c1 = np.array(c1_list)
                dists = self._compute_view_dist(Xs[1], cur_c1, 1)
                min_dists = np.min(dists, axis=1)
                probs = min_dists**2
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs /= prob_sum
                    next_idx = rng.choice(n_samples, p=probs)
                else:
                    next_idx = rng.randint(n_samples)
                indices.append(next_idx)
                c1_list.append(Xs[1][next_idx])

            c0 = Xs[0][indices].copy()
            c1 = np.array(c1_list)
        else:
            raise ValueError(f"Unknown init method: {self.init}")

        if self.view_metrics[0].lower() == "cosine":
            c0 = normalize(c0)
        if self.view_metrics[1].lower() == "cosine":
            c1 = normalize(c1)

        return [c0, c1]

    def _em_step(
        self,
        X: np.ndarray,
        partition: np.ndarray,
        centroids: np.ndarray,
        view_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Executes one expectation-maximization step on a single view."""
        n_samples = X.shape[0]
        new_centers = []

        for cl in range(self.n_clusters):
            mask = partition == cl
            if np.sum(mask) == 0:
                new_centers.append(centroids[cl])
            else:
                cent = np.mean(X[mask], axis=0)
                new_centers.append(cent)

        new_centers_arr = np.vstack(new_centers)
        if self.view_metrics[view_idx].lower() == "cosine":
            new_centers_arr = normalize(new_centers_arr)

        distances = self._compute_view_dist(X, new_centers_arr, view_idx)
        new_parts = np.argmin(distances, axis=1).flatten()
        min_dists = distances[np.arange(n_samples), new_parts]
        o_funct = float(np.sum(min_dists))

        return new_parts, new_centers_arr, o_funct

    def _one_init(
        self, Xs: List[np.ndarray], seed: Optional[int]
    ) -> Tuple[float, List[np.ndarray]]:
        """Runs the alternating EM optimization for a single initialization seed."""
        rng = np.random.RandomState(seed)
        centroids = self._init_centroids(Xs, rng)

        # Initial partition derived from View 1
        d1 = self._compute_view_dist(Xs[1], centroids[1], 1)
        parts = np.argmin(d1, axis=1).flatten()
        partitions = [None, parts]
        objective = [np.inf, np.inf]
        o_funct = [None, None]
        iter_stall = [0, 0]
        iter_num = 0
        max_iter = self.max_iter if self.max_iter is not None else np.inf

        while max(iter_stall) < self.patience and iter_num < max_iter:
            for vi in range(2):
                pre_view = (iter_num + 1) % 2
                partitions[vi], centroids[vi], o_funct[vi] = self._em_step(
                    Xs[vi], partitions[pre_view], centroids[vi], vi
                )
            iter_num += 1

            for view in range(2):
                if objective[view] - o_funct[view] > self.tol * np.abs(objective[view]):
                    objective[view] = o_funct[view]
                    iter_stall[view] = 0
                else:
                    iter_stall[view] += 1

        inertia = float(np.sum(objective))
        return inertia, centroids

    def _final_centroids(self, Xs: List[np.ndarray], centroids: List[np.ndarray]):
        """Derives final consensus centroids from mutually agreed sample partitions."""
        v1_consensus = []
        v2_consensus = []

        v1_distances = self._compute_view_dist(Xs[0], centroids[0], 0)
        v1_partitions = np.argmin(v1_distances, axis=1).flatten()
        v2_distances = self._compute_view_dist(Xs[1], centroids[1], 1)
        v2_partitions = np.argmin(v2_distances, axis=1).flatten()

        for clust in range(self.n_clusters):
            part_indices = (v1_partitions == clust) & (v2_partitions == clust)
            if np.sum(part_indices) > 0:
                cent1 = np.mean(Xs[0][part_indices], axis=0)
                v1_consensus.append(cent1)
                cent2 = np.mean(Xs[1][part_indices], axis=0)
                v2_consensus.append(cent2)

        if len(v1_consensus) == 0:
            warnings.warn(
                "No distinct cluster centroids have been found.", ConvergenceWarning
            )
            self.centroids_ = [centroids[0].copy(), centroids[1].copy()]
        else:
            c1 = np.vstack(v1_consensus)
            c2 = np.vstack(v2_consensus)
            if self.view_metrics[0].lower() == "cosine":
                c1 = normalize(c1)
            if self.view_metrics[1].lower() == "cosine":
                c2 = normalize(c2)
            self.centroids_ = [c1, c2]

            if self.centroids_[0].shape[0] < self.n_clusters:
                warnings.warn(
                    f"Distinct cluster centroids ({self.centroids_[0].shape[0]}) "
                    f"found is smaller than n_clusters ({self.n_clusters}).",
                    ConvergenceWarning,
                )
            self.n_clusters = self.centroids_[0].shape[0]

    def fit(self, Xs: Sequence[Any], y=None):
        """Fit multi-view cluster centroids to the multiple data views."""
        Xs_clean = self._preprocess_data(Xs)

        rng_base = np.random.RandomState(self.random_state)
        seeds = [rng_base.randint(0, 2**31 - 1) for _ in range(self.n_init)]

        run_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._one_init)(Xs_clean, s) for s in seeds
        )

        inertias, centroids_list = zip(*run_results)
        # Select seed with minimal inertia (best convergence)
        best_ind = int(np.argmin(inertias))

        self.inertia_ = float(inertias[best_ind])
        self._final_centroids(Xs_clean, centroids_list[best_ind])
        self.labels_ = self.predict(Xs_clean)
        return self

    def predict(self, Xs: Sequence[Any]) -> np.ndarray:
        """Predict cluster assignments using weighted combined view distances."""
        Xs_clean = self._preprocess_data(Xs)

        if self.centroids_ is None or self.centroids_[0] is None:
            raise NotFittedError(
                "This DecoupledMultiviewKMeans instance is not fitted yet."
            )

        dist0 = self._compute_view_dist(Xs_clean[0], self.centroids_[0], 0)
        dist1 = self._compute_view_dist(Xs_clean[1], self.centroids_[1], 1)

        total_dist = self.view_weights[0] * dist0 + self.view_weights[1] * dist1
        return np.argmin(total_dist, axis=1).flatten()
