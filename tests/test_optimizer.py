import pytest
from src.optimizer import Optimizer

# A minimal experiment config for testing
MOCK_EXPERIMENT_CONFIG = {"experiment": {}}

def create_optimizer(model_config):
    """Helper function to create a dummy Optimizer instance for testing."""
    return Optimizer(
        texts=[],
        embeddings=None,
        scaled_metadata=None,
        model_config=model_config,
        experiment_config=MOCK_EXPERIMENT_CONFIG
    )

def test_config_with_no_search_space():
    """
    Tests that a single config is returned when teh config doesn't define a hyperparameter
    search space (i.e., no lists of values).
    """
    model_config = {
        "id": "test",
        "clustering": {
            "params": {
                "n_clusters": 50
            }
        }
    }
    optimizer = create_optimizer(model_config)
    combinations = optimizer._generate_hyperparameter_combinations()
    
    assert len(combinations) == 1
    assert combinations[0][0] == model_config
    assert combinations[0][1] == {}

def test_single_hyperparameter():
    """
    Tests generation with a single hyperparameter list.
    """
    model_config = {
        "id": "test",
        "clustering": {
            "params": {
                "n_clusters": [10, 20, 30]
            }
        }
    }
    optimizer = create_optimizer(model_config)
    combinations = optimizer._generate_hyperparameter_combinations()
    
    assert len(combinations) == 3
    
    # Check generated configs
    assert combinations[0][0]["clustering"]["params"]["n_clusters"] == 10
    assert combinations[1][0]["clustering"]["params"]["n_clusters"] == 20
    assert combinations[2][0]["clustering"]["params"]["n_clusters"] == 30
    
    # Check varied params dict
    assert combinations[0][1] == {"clustering.params.n_clusters": 10}
    assert combinations[1][1] == {"clustering.params.n_clusters": 20}
    assert combinations[2][1] == {"clustering.params.n_clusters": 30}

def test_multiple_hyperparameters():
    """
    Tests the cartesian product of multiple hyperparameter lists.
    """
    model_config = {
        "id": "test",
        "dimensionality_reduction": {
            "params": {
                "n_components": [5, 10]
            }
        },
        "clustering": {
            "params": {
                "n_clusters": [100, 200]
            }
        }
    }
    optimizer = create_optimizer(model_config)
    combinations = optimizer._generate_hyperparameter_combinations()
    
    assert len(combinations) == 4 # 2 * 2
    
    # Check that all combinations are present
    expected_configs = [
        (5, 100),
        (5, 200),
        (10, 100),
        (10, 200)
    ]
    generated_configs = [
        (c[0]["dimensionality_reduction"]["params"]["n_components"], c[0]["clustering"]["params"]["n_clusters"])
        for c in combinations
    ]
    assert sorted(generated_configs) == sorted(expected_configs)

    # Check one of the varied_params dicts
    assert {"dimensionality_reduction.params.n_components": 5, "clustering.params.n_clusters": 200} in [c[1] for c in combinations]

def test_string_list_is_ignored():
    """
    Tests that a list of strings is correctly ignored and not treated as a
    hyperparameter to be varied.
    """
    model_config = {
        "id": "test",
        "representation_model": {
            "type": "KeyBERT",
            "params": {
                # This is a valid parameter value, not a list to iterate over
                "stop_words": ["english", "custom"]
            }
        },
        "clustering": {
            "params": {
                "n_clusters": [10, 20]
            }
        }
    }
    optimizer = create_optimizer(model_config)
    combinations = optimizer._generate_hyperparameter_combinations()

    # Should only generate combinations for n_clusters
    assert len(combinations) == 2
    assert combinations[0][0]["representation_model"]["params"]["stop_words"] == ["english", "custom"]
    assert combinations[1][0]["representation_model"]["params"]["stop_words"] == ["english", "custom"]
    assert combinations[0][0]["clustering"]["params"]["n_clusters"] == 10
    assert combinations[1][0]["clustering"]["params"]["n_clusters"] == 20
