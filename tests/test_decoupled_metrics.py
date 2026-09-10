"""Unit tests for decoupled multi-view distance metrics and backward compatibility."""

import numpy as np
from mvlearn.cluster import MultiviewKMeans

from src.decoupled_kmeans import DecoupledMultiviewKMeans
from src.models import create_topic_model_instance, get_algorithm
from src.mvc_wrapper import MVCWrapper
from src.optimizer import generate_hyperparameter_combinations
from src.utils import load_config


def test_mvc_wrapper_backward_compatibility():
    """Verify MVCWrapper defaults normalize_text_view to False."""
    np.random.seed(42)
    X = np.random.randn(20, 5) * 10
    metadata = np.random.rand(20, 3)

    # Instantiate with default
    wrapper_default = MVCWrapper(
        model=MultiviewKMeans(n_clusters=2, random_state=42), metadata=metadata
    )
    assert not wrapper_default.normalize_text_view

    views_default = wrapper_default._prepare_views(X)
    np.testing.assert_array_equal(views_default[0], X)
    np.testing.assert_array_equal(views_default[1], metadata)

    # Explicit False
    wrapper_false = MVCWrapper(
        model=MultiviewKMeans(n_clusters=2, random_state=42),
        metadata=metadata,
        normalize_text_view=False,
    )
    views_false = wrapper_false._prepare_views(X)
    np.testing.assert_array_equal(views_false[0], X)


def test_mvc_wrapper_l2_normalization():
    """Verify MVCWrapper normalizes View 0 while leaving View 1 untouched."""
    np.random.seed(42)
    X = np.random.randn(20, 5) * 10
    metadata = np.random.rand(20, 3)

    wrapper = MVCWrapper(
        model=MultiviewKMeans(n_clusters=2, random_state=42),
        metadata=metadata,
        normalize_text_view=True,
    )
    assert wrapper.normalize_text_view

    views = wrapper._prepare_views(X)
    # View 0 should have unit norm
    norms = np.linalg.norm(views[0], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # View 1 should be completely untouched
    np.testing.assert_array_equal(views[1], metadata)


def test_get_algorithm_backward_compatibility():
    """Verify existing multi-view configs without normalize_text_view work."""
    metadata = np.random.rand(20, 3)

    # Existing multi_view_k_means config
    conf_kmeans = {"type": "multi_view_k_means", "params": {"n_clusters": 3}}
    algo_kmeans = get_algorithm(conf_kmeans, metadata=metadata, random_state=42)
    assert isinstance(algo_kmeans, MVCWrapper)
    assert not algo_kmeans.normalize_text_view
    assert isinstance(algo_kmeans.model, MultiviewKMeans)

    # Existing multi_view_spectral_clustering config
    conf_spectral = {
        "type": "multi_view_spectral_clustering",
        "params": {"n_clusters": 3},
    }
    algo_spectral = get_algorithm(conf_spectral, metadata=metadata, random_state=42)
    assert isinstance(algo_spectral, MVCWrapper)
    assert not algo_spectral.normalize_text_view


def test_get_algorithm_new_options():
    """Verify factory parses normalize_text_view and decoupled_multi_view_k_means."""
    metadata = np.random.rand(20, 3)

    # Approach 1: normalize_text_view=True
    conf_norm = {
        "type": "multi_view_k_means",
        "params": {"normalize_text_view": True, "n_clusters": 3},
    }
    algo_norm = get_algorithm(conf_norm, metadata=metadata, random_state=42)
    assert isinstance(algo_norm, MVCWrapper)
    assert algo_norm.normalize_text_view

    # Approach 2: decoupled_multi_view_k_means
    conf_decoupled = {
        "type": "decoupled_multi_view_k_means",
        "params": {
            "view_metrics": ["cosine", "euclidean"],
            "view_weights": [1.0, 1.0],
            "n_clusters": 3,
        },
    }
    algo_decoupled = get_algorithm(conf_decoupled, metadata=metadata, random_state=42)
    assert isinstance(algo_decoupled, MVCWrapper)
    assert isinstance(algo_decoupled.model, DecoupledMultiviewKMeans)
    assert algo_decoupled.model.view_metrics == ["cosine", "euclidean"]


def test_decoupled_multiview_kmeans_fit_predict():
    """Verify DecoupledMultiviewKMeans fits and predicts with decoupled metrics."""
    np.random.seed(42)
    v0 = np.random.randn(40, 8)
    v1 = np.random.rand(40, 3)
    Xs = [v0, v1]

    model = DecoupledMultiviewKMeans(
        n_clusters=3,
        random_state=42,
        view_metrics=("cosine", "euclidean"),
        view_weights=(1.0, 1.0),
        n_init=3,
    )
    model.fit(Xs)

    assert model.centroids_ is not None
    assert len(model.centroids_) == 2
    # View 0 centroids should be unit normalized for cosine metric
    c0_norms = np.linalg.norm(model.centroids_[0], axis=1)
    np.testing.assert_allclose(c0_norms, 1.0, atol=1e-5)

    # Labels should be valid
    labels = model.predict(Xs)
    assert len(labels) == 40
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_decoupled_multiview_spectral_fit_predict():
    """Verify DecoupledMultiviewSpectralClustering fits and predicts."""
    from src.decoupled_spectral import DecoupledMultiviewSpectralClustering

    np.random.seed(42)
    v0 = np.random.randn(30, 8)
    v1 = np.random.rand(30, 3)
    Xs = [v0, v1]

    model = DecoupledMultiviewSpectralClustering(
        n_clusters=3,
        random_state=42,
        view_affinities=("cosine", "rbf"),
        max_iter=3,
        n_init=2,
    )
    model.fit(Xs)

    assert model.labels_ is not None
    assert len(model.labels_) == 30
    assert set(np.unique(model.labels_)).issubset({0, 1, 2})


def test_yelp_new_yaml_configs_instantiation():
    """Verify all new Yelp experiment YAML configs load and instantiate cleanly."""
    import pathlib

    exp_dir = pathlib.Path("experiments")
    metadata = np.random.rand(20, 4)

    for exp_rel in [
        "yelp/yelp_standard_mv_k_means_l2_norm.yaml",
        "yelp/yelp_standard_decoupled_mv_k_means.yaml",
        "yelp/yelp_standard_mv_spectral_l2_norm.yaml",
        "yelp/yelp_standard_decoupled_mv_spectral.yaml",
    ]:
        cfg = load_config(exp_rel, exp_dir)
        assert "experiment" in cfg
        assert "model" in cfg

        model_cfg = cfg["model"]
        combinations = generate_hyperparameter_combinations(model_cfg)
        assert len(combinations) == 5  # 5 topic counts: 10, 20, 30, 40, 50

        for combo, _ in combinations:
            inst = create_topic_model_instance(
                model_config=combo,
                scaled_metadata=metadata,
                random_state=36201624,
            )
            assert inst is not None
            assert hasattr(inst, "hdbscan_model")
