from typing import Any, Optional, Union

import mvlearn.cluster as mvcluster
import numpy as np
import polars as pl
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from src.mvc_wrapper import MVCWrapper


def create_topic_model_instance(
    model_config: dict,
    scaled_metadata: Optional[Union[pl.DataFrame, np.ndarray]],
    random_state: int,
    n_clusters: Optional[int] = None,
    remove_rep_stopwords: bool = False,
) -> Any:
    """
    Factory function that creates a topic model instance (BERTopic or TriTopic)
    based on the provided configuration dictionary.
    """
    model_type = model_config.get("type") or model_config.get("model_type")
    if model_type == "tritopic":
        return create_tritopic_instance(
            model_config=model_config,
            random_state=random_state,
            n_clusters=n_clusters,
            remove_rep_stopwords=remove_rep_stopwords,
        )

    return create_bertopic_instance(
        model_config=model_config,
        scaled_metadata=scaled_metadata,
        random_state=random_state,
        n_clusters=n_clusters,
        remove_rep_stopwords=remove_rep_stopwords,
    )


def create_tritopic_instance(
    model_config: dict,
    random_state: int,
    n_clusters: Optional[int] = None,
    remove_rep_stopwords: bool = False,
) -> Any:
    """
    Factory function that builds a TriTopic instance from a configuration dictionary.
    """
    from tritopic import TriTopic, TriTopicConfig

    params = model_config.get("params") or {}
    params = params.copy()

    if "random_state" not in params:
        params["random_state"] = random_state

    # Standardize topic count across models
    if n_clusters is not None:
        n_topics = n_clusters
        params.pop("n_clusters", None)
        params.pop("n_topics", None)
    elif "n_clusters" in params:
        n_topics = params.pop("n_clusters")
        params.pop("n_topics", None)
    else:
        n_topics = params.pop("n_topics", "auto")

    # Default use_metadata_view to True so metadata is incorporated
    # into the tri-modal graph
    if "use_metadata_view" not in params:
        params["use_metadata_view"] = True

    config_obj = TriTopicConfig(**params)
    return TriTopic(config=config_obj, n_topics=n_topics)


def create_bertopic_instance(
    model_config: dict,
    scaled_metadata: Optional[Union[pl.DataFrame, np.ndarray]],
    random_state: int,
    n_clusters: Optional[int] = None,
    remove_rep_stopwords: bool = False,
) -> BERTopic:
    """
    Factory function that builds a BERTopic instance from a configuration dictionary.

    Args:
        model_config: Dictionary containing model hyperparameters
            (clustering, dim reduction).
        scaled_metadata: Metadata DataFrame or array required by the custom algorithms.
        random_state: Seed for reproducibility.
        n_clusters: Optional integer to force a specific number of topics
            (used for non-baseline models).
        remove_rep_stopwords: If True, removes English stop words from
            the c-TF-IDF topic word representations using CountVectorizer.
            Note: This only filters representation topic words and does NOT
            modify original document texts or document embeddings.

    Returns:
        An unfitted BERTopic instance.
    """
    # Instantiate Dimensionality Reduction
    umap_model = get_algorithm(
        model_config["dimensionality_reduction"],
        metadata=scaled_metadata,
        random_state=random_state,
    )

    # Instantiate Clustering
    use_baseline_topics: bool = model_config.get("use_baseline_n_topics", False)
    if use_baseline_topics:
        hdbscan_model = get_algorithm(
            model_config["clustering"],
            metadata=scaled_metadata,
            random_state=random_state,
            n_clusters=n_clusters,
        )
    else:
        hdbscan_model = get_algorithm(
            model_config["clustering"],
            metadata=scaled_metadata,
            random_state=random_state,
        )

    # Extract BERTopic parameters
    bertopic_config = model_config.get("bertopic") or {}
    bertopic_params = bertopic_config.get("params") or {}
    # Copy parameters to avoid modifying the original config dict
    bertopic_params = bertopic_params.copy()
    if "top_n_words" not in bertopic_params:
        bertopic_params["top_n_words"] = 50

    # Check if representation stop words flag is in config or explicitly passed
    config_stop_words = bertopic_params.pop("remove_rep_stopwords", False)
    should_remove_stop_words = remove_rep_stopwords or config_stop_words

    if should_remove_stop_words and "vectorizer_model" not in bertopic_params:
        bertopic_params["vectorizer_model"] = CountVectorizer(stop_words="english")

    # Return the assembled object
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        **bertopic_params,
    )


def get_algorithm(
    config: dict,
    metadata: Optional[Union[pl.DataFrame, np.ndarray, Any]],
    random_state: int,
    n_clusters: Optional[int] = None,
):
    algo_type = config["type"]
    params = config.get("params") or {}

    if n_clusters:
        params["n_clusters"] = n_clusters

    if algo_type == "umap":
        from umap import UMAP

        return UMAP(random_state=random_state, **params)

    elif algo_type == "append_umap":
        from src.append_umap import AppendUMAP

        if metadata is None:
            raise ValueError("Metadata array is null")
        return AppendUMAP(metadata=metadata, random_state=random_state, **params)

    elif algo_type == "aligned_umap":
        from umap import AlignedUMAP

        from src.mvc_wrapper import AlignedUMAPWrapper

        if metadata is None:
            raise ValueError("Metadata array is null")

        aligned_umap = AlignedUMAP(random_state=random_state, **params)
        return AlignedUMAPWrapper(model=aligned_umap, metadata=metadata)

    elif algo_type == "pca":
        from sklearn.decomposition import PCA

        return PCA(random_state=random_state, **params)

    elif algo_type == "hdbscan":
        from hdbscan import HDBSCAN

        return HDBSCAN(**params)  # No random state parameter

    elif algo_type == "k_means":
        from sklearn.cluster import KMeans

        return KMeans(random_state=random_state, **params)

    elif algo_type == "spectral_clustering":
        from sklearn.cluster import SpectralClustering

        return SpectralClustering(random_state=random_state, **params)

    elif algo_type == "multi_view_spectral_clustering":
        if metadata is None:
            raise ValueError("Metadata array is null")

        cluster_model = mvcluster.MultiviewSpectralClustering(
            random_state=random_state, **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)

    elif algo_type == "co_regularized_multi_view_spectral_clustering":
        if metadata is None:
            raise ValueError("Metadata array is null")

        cluster_model = mvcluster.MultiviewCoRegSpectralClustering(
            random_state=random_state, **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)

    elif algo_type == "multi_view_k_means":
        if metadata is None:
            raise ValueError("Metadata array is null")

        cluster_model = mvcluster.MultiviewKMeans(random_state=random_state, **params)
        return MVCWrapper(model=cluster_model, metadata=metadata)

    elif algo_type == "multi_view_spherical_k_means":
        if metadata is None:
            raise ValueError("Metadata array is null")

        cluster_model = mvcluster.MultiviewSphericalKMeans(
            random_state=random_state, **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)

    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
