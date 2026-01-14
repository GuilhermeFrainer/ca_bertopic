import numpy as np
import mvlearn.cluster as mvcluster
from bertopic import BERTopic

from typing import Optional

from src.mvc_wrapper import MVCWrapper


def create_bertopic_instance(
    model_config: dict, 
    scaled_metadata: np.ndarray, 
    random_state: int,
    n_clusters: Optional[int] = None
) -> BERTopic:
    """
    Factory function that builds a BERTopic instance from a configuration dictionary.
    
    Args:
        model_config: Dictionary containing model hyperparameters (clustering, dim reduction).
        scaled_metadata: Metadata array required by the custom algorithms.
        random_state: Seed for reproducibility.
        n_clusters: Optional integer to force a specific number of topics (used for non-baseline models).

    Returns:
        An unfitted BERTopic instance.
    """
    # Instantiate Dimensionality Reduction
    umap_model = get_algorithm(
        model_config["dimensionality_reduction"],
        metadata=scaled_metadata,
        random_state=random_state
    )

    # Instantiate Clustering
    use_baseline_topics: bool = model_config.get("use_baseline_n_topics", False)
    if use_baseline_topics:
        hdbscan_model = get_algorithm(
            model_config["clustering"],
            metadata=scaled_metadata,
            random_state=random_state,
            n_clusters=n_clusters 
        )
    else:
        hdbscan_model = get_algorithm(
            model_config["clustering"],
            metadata=scaled_metadata,
            random_state=random_state
        )
    
    # Return the assembled object
    return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)


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
        from umap import UMAP
        return UMAP(random_state=random_state, **params)
    
    elif algo_type == 'pca':
        from sklearn.decomposition import PCA
        return PCA(random_state=random_state, **params)
    
    elif algo_type == 'hdbscan':
        from hdbscan import HDBSCAN
        return HDBSCAN(**params) # No random state parameter
    
    elif algo_type == 'k_means':
        from sklearn.cluster import KMeans
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

