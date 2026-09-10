"""
Decoupled Multi-View Spectral Clustering.

Allows specifying independent affinity kernels for each view (e.g. Cosine similarity
for textual embeddings in View 0 and Euclidean RBF for continuous metadata in View 1).
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
from mvlearn.cluster import MultiviewSpectralClustering
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class DecoupledMultiviewSpectralClustering(MultiviewSpectralClustering):
    """
    Multi-View Spectral Clustering with decoupled per-view affinity metrics.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to extract.
    random_state : int, optional, default=None
        Random state for reproducibility in the final KMeans step.
    info_view : int, optional, default=None
        Designated view index to use for spectral representation if desired.
    max_iter : int, default=10
        Maximum number of co-training iterations between views.
    n_init : int, default=10
        Number of KMeans initializations.
    view_affinities : Union[str, Sequence[str]], default=('cosine', 'rbf')
        Affinity kernel to construct for each view.
        Supported options: 'cosine', 'cosine_nn', 'rbf', 'nearest_neighbors', 'poly'.
    gamma : float, optional, default=None
        Kernel bandwidth coefficient for RBF kernel.
    n_neighbors : int, default=10
        Number of neighbors for nearest neighbor affinity constructions.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        random_state: Optional[int] = None,
        info_view: Optional[int] = None,
        max_iter: int = 10,
        n_init: int = 10,
        view_affinities: Union[str, Sequence[str]] = ("cosine", "rbf"),
        gamma: Optional[float] = None,
        n_neighbors: int = 10,
    ):
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            info_view=info_view,
            max_iter=max_iter,
            n_init=n_init,
            affinity="rbf",  # Placeholder to satisfy superclass validation
            gamma=gamma,
            n_neighbors=n_neighbors,
        )
        if isinstance(view_affinities, str):
            self.view_affinities = [a.strip() for a in view_affinities.split(",")]
        else:
            self.view_affinities = list(view_affinities)

    def _affinity_mat_view(self, X: np.ndarray, view_idx: int) -> np.ndarray:
        """Computes affinity matrix according to the specific view's metric."""
        affinity = self.view_affinities[view_idx].lower()

        if affinity == "cosine":
            X_norm = normalize(X.astype(float))
            sims = np.maximum(0.0, X_norm @ X_norm.T)
            np.fill_diagonal(sims, 1.0)
            return sims

        elif affinity in ("cosine_nn", "cosine_nearest_neighbors"):
            neighbor = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")
            neighbor.fit(X)
            dist_graph = neighbor.kneighbors_graph(X, mode="distance").toarray()
            sims = np.where(dist_graph > 0, np.maximum(0.0, 1.0 - dist_graph), 0.0)
            np.fill_diagonal(sims, 1.0)
            return (sims + sims.T) / 2.0

        elif affinity in ("nearest_neighbors", "knn"):
            neighbor = NearestNeighbors(
                n_neighbors=self.n_neighbors, metric="euclidean"
            )
            neighbor.fit(X)
            sims = neighbor.kneighbors_graph(X, mode="connectivity").toarray()
            return (sims + sims.T) / 2.0

        elif affinity == "rbf":
            gamma = self.gamma
            if gamma is None:
                distances = cdist(X, X)
                med = np.median(distances)
                gamma = 1.0 / (2.0 * (med**2)) if med > 0 else 1.0
            return rbf_kernel(X, gamma=gamma)

        elif affinity == "poly":
            return polynomial_kernel(X, gamma=self.gamma)

        else:
            raise ValueError(f"Unknown affinity '{affinity}' for view {view_idx}")

    def fit(self, Xs: Sequence[Any], y=None):
        """Fit multi-view spectral clustering model."""
        Xs_checked = self._param_checks(Xs)
        if len(self.view_affinities) != self._n_views:
            raise ValueError(
                f"view_affinities length ({len(self.view_affinities)}) "
                f"must match number of views ({self._n_views})"
            )

        sims = [self._affinity_mat_view(X, i) for i, X in enumerate(Xs_checked)]
        U_mats = [self._compute_eigs(sim) for sim in sims]

        for _ in range(self.max_iter):
            eig_sums = [u_mat @ np.transpose(u_mat) for u_mat in U_mats]
            U_sum = np.sum(np.array(eig_sums), axis=0)
            new_sims = []

            for view in range(self._n_views):
                mat1 = sims[view] @ (U_sum - eig_sums[view])
                mat1 = (mat1 + np.transpose(mat1)) / 2.0
                new_sims.append(mat1)
                U_mats = [self._compute_eigs(sim) for sim in new_sims]

        for view in range(self._n_views):
            U_norm = np.linalg.norm(U_mats[view], axis=1).reshape((-1, 1))
            U_norm[U_norm == 0] = 1.0
            U_mats[view] /= U_norm

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )

        if self.info_view is not None:
            self.embedding_ = U_mats[self.info_view]
            self.labels_ = kmeans.fit_predict(self.embedding_)
        else:
            self.embedding_ = np.hstack(U_mats)
            self.labels_ = kmeans.fit_predict(self.embedding_)

        return self
