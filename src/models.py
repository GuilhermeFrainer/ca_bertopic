import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import mvlearn.cluster as mvcluster

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

    if n_clusters:
        params["n_clusters"] = n_clusters

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

        cluster_model = mvcluster.MultiviewSpectralClustering(
            random_state=random_state,
            **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)
    elif algo_type == 'co_regularized_multi_view_spectral_clustering':
        if metadata is None:
            raise ValueError("Metadata array is null")

        cluster_model = mvcluster.MultiviewCoRegSpectralClustering(
            random_state=random_state,
            **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)
    elif algo_type == 'multi_view_k_means':
        if metadata is None:
            raise ValueError("Metadata array is null")

        cluster_model = mvcluster.MultiviewKMeans(
            random_state=random_state,
            **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)
    elif algo_type == 'multi_view_spherical_k_means':
        if metadata is None:
            raise ValueError("Metadata array is null")

        cluster_model = mvcluster.MultiviewSphericalKMeans(
            random_state=random_state,
            **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
    
