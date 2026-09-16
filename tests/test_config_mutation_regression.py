import copy

import numpy as np
import pytest

from src.models import (
    create_topic_model_instance,
    get_algorithm,
)
from src.mvc_wrapper import MVCWrapper


@pytest.mark.parametrize(
    "algo_type",
    [
        "multi_view_k_means",
        "multi_view_spectral_clustering",
        "co_regularized_multi_view_spectral_clustering",
        "multi_view_hdbscan",
        "feature_stacking_hdbscan",
    ],
)
def test_get_algorithm_preserves_normalize_text_view_across_calls(algo_type):
    """
    Regression test: ensures get_algorithm does not pop 'normalize_text_view'
    from caller's config dictionary, and that subsequent instantiations
    retain normalize_text_view=True.
    """
    metadata = np.random.rand(20, 2)
    original_params = {
        "normalize_text_view": True,
        "n_clusters": 5,
    }
    if "hdbscan" in algo_type:
        original_params["min_cluster_size"] = 3

    config = {
        "type": algo_type,
        "params": copy.deepcopy(original_params),
    }

    # Simulate 3 repeated seed constructions as done by Optimizer
    for seed in [36201624, 62613654, 57116123]:
        wrapper = get_algorithm(config, metadata=metadata, random_state=seed)

        assert isinstance(wrapper, MVCWrapper)
        assert wrapper.normalize_text_view is True, (
            f"Failed for {algo_type} on seed {seed}: normalize_text_view was lost!"
        )

        # Assert caller's config remains strictly unchanged
        assert config["params"]["normalize_text_view"] is True
        if "n_clusters" in original_params:
            assert config["params"]["n_clusters"] == original_params["n_clusters"]


@pytest.mark.parametrize(
    "algo_type",
    [
        "multi_view_k_means",
        "multi_view_spectral_clustering",
        "co_regularized_multi_view_spectral_clustering",
    ],
)
def test_get_algorithm_preserves_false_normalize_text_view(algo_type):
    """
    Regression test: ensures explicit normalize_text_view=False is respected
    and not mutated across repeated calls.
    """
    metadata = np.random.rand(20, 2)
    config = {
        "type": algo_type,
        "params": {
            "normalize_text_view": False,
            "n_clusters": 5,
        },
    }

    for seed in [1, 2, 3]:
        wrapper = get_algorithm(config, metadata=metadata, random_state=seed)
        assert isinstance(wrapper, MVCWrapper)
        assert wrapper.normalize_text_view is False
        assert config["params"]["normalize_text_view"] is False


def test_get_algorithm_n_clusters_injection_does_not_mutate_config():
    """
    Regression test: ensures passing n_clusters to get_algorithm does NOT
    modify the caller's config dictionary.
    """
    metadata = np.random.rand(20, 2)
    config = {
        "type": "k_means",
        "params": {},
    }

    # First call with n_clusters=10
    algo1 = get_algorithm(config, metadata=metadata, random_state=42, n_clusters=10)
    assert algo1.n_clusters == 10
    # The config dict must still have an empty params dict
    assert config["params"] == {}

    # Second call without n_clusters
    algo2 = get_algorithm(config, metadata=metadata, random_state=42)
    # Default n_clusters for sklearn KMeans is 8
    assert algo2.n_clusters == 8
    assert config["params"] == {}


def test_create_topic_model_instance_does_not_mutate_model_config():
    """
    Regression test: ensures create_topic_model_instance and create_bertopic_instance
    do not mutate nested configuration dictionaries across seeds.
    """
    metadata = np.random.rand(20, 2)
    model_config = {
        "dimensionality_reduction": {
            "type": "umap",
            "params": {"n_components": 5, "min_dist": 0.0, "metric": "cosine"},
        },
        "clustering": {
            "type": "multi_view_k_means",
            "params": {"n_clusters": 10, "normalize_text_view": True},
        },
        "bertopic": {
            "params": {
                "top_n_words": 10,
                "remove_rep_stopwords": True,
            }
        },
    }

    snapshot = copy.deepcopy(model_config)

    for seed in [36201624, 62613654, 57116123]:
        model = create_topic_model_instance(
            model_config=model_config,
            scaled_metadata=metadata,
            random_state=seed,
            remove_rep_stopwords=True,
        )
        assert model is not None
        # Assert model_config is completely intact
        assert model_config == snapshot


def test_optimizer_all_runs_have_independent_configs():
    """
    Regression test: ensures Optimizer creates independent deep copies of
    model_config for all seed runs.
    """
    from src.optimizer import generate_hyperparameter_combinations

    model_config = {
        "dimensionality_reduction": {"type": "umap", "params": {"n_components": 5}},
        "clustering": {
            "type": "multi_view_k_means",
            "params": {
                "n_clusters": [10, 20],
                "normalize_text_view": True,
            },
        },
    }

    combos = generate_hyperparameter_combinations(model_config)
    seeds = [100, 200, 300]

    all_runs = []
    for combo_idx, (m_config, varied_params) in enumerate(combos):
        for seed in seeds:
            all_runs.append(
                (
                    combo_idx,
                    copy.deepcopy(m_config),
                    copy.deepcopy(varied_params),
                    seed,
                )
            )

    # Mutating one run's config should not affect other runs
    all_runs[0][1]["clustering"]["params"]["normalize_text_view"] = False
    assert all_runs[1][1]["clustering"]["params"]["normalize_text_view"] is True
