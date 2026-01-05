import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mvlearn.cluster import MultiviewSpectralClustering

from typing import Optional

from src.mvc_wrapper import MVCWrapper


def get_algorithm(
    config: dict,
    metadata: Optional[np.ndarray],
    random_state: int,
    n_clusters: Optional[int] = None
):
    algo_type = config["type"]
    params = config.get("params") or {}

    if algo_type == 'umap':
        return UMAP(random_state=random_state, **params)
    elif algo_type == 'pca':
        return PCA(random_state=random_state, **params)
    elif algo_type == 'hdbscan':
        return HDBSCAN(**params) # No random state parameter
    elif algo_type == 'k_means':
        return KMeans(random_state=random_state, **params)
    elif algo_type == 'multi_view_spectral_clustering':
        if metadata is None:
            raise ValueError("Metadata array is null")

        if n_clusters and params["n_clusters"] == "baseline":
            params["n_clusters"] = n_clusters

        cluster_model = MultiviewSpectralClustering(
            random_state=random_state,
            **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)
        
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")