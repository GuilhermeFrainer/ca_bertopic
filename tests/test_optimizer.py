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

def test_range_hyperparameter():
    """
    Tests generation with a range hyperparameter.
    """
    model_config = {
        "id": "test",
        "clustering": {
            "params": {
                "n_clusters": {
                    "start": 10,
                    "stop": 31,
                    "step": 10
                }
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

def test_float_range_hyperparameter():
    """
    Tests generation with a float range hyperparameter.
    """
    model_config = {
        "id": "test",
        "dimensionality_reduction": {
            "params": {
                "some_float": {
                    "start": 0.1,
                    "stop": 0.31,
                    "step": 0.1
                }
            }
        }
    }
    optimizer = create_optimizer(model_config)
    combinations = optimizer._generate_hyperparameter_combinations()
    
    assert len(combinations) == 3 # 0.1, 0.2, 0.3
    
    # Check generated configs and varied params dict
    # Note: due to float precision, it's better to check if they are close
    generated_values = [c[0]["dimensionality_reduction"]["params"]["some_float"] for c in combinations]
    expected_values = [0.1, 0.2, 0.3]
    assert all(pytest.approx(g) == e for g, e in zip(generated_values, expected_values))

    varied_params = [c[1] for c in combinations]
    expected_varied = [
        {"dimensionality_reduction.params.some_float": 0.1},
        {"dimensionality_reduction.params.some_float": 0.2},
        {"dimensionality_reduction.params.some_float": 0.3},
    ]
    assert all(pytest.approx(v["dimensionality_reduction.params.some_float"]) == e["dimensionality_reduction.params.some_float"] for v, e in zip(varied_params, expected_varied))


def test_mixed_hyperparameters():
    """
    Tests a mix of list and range hyperparameters.
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
                "n_clusters": {
                    "start": 100,
                    "stop": 201,
                    "step": 100
                }
            }
        }
    }
    optimizer = create_optimizer(model_config)
    combinations = optimizer._generate_hyperparameter_combinations()
    
    assert len(combinations) == 4 # 2 * 2
    
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


def test_determinism_via_sorting():
    """
    Tests that the order of hyperparameter combinations is deterministic
    by ensuring it doesn't depend on the dictionary order of the config.
    """
    # Create two configs with the same keys but different insertion order
    config_a = {
        "dimensionality_reduction": {
            "params": {
                "n_components": [5, 10],
                "some_param": [1, 2]
            }
        },
        "clustering": {
            "params": {
                "n_clusters": [100, 200]
            }
        }
    }
    
    config_b = {
        "dimensionality_reduction": {
            "params": {
                "some_param": [1, 2],
                "n_components": [5, 10]
            }
        },
        "clustering": {
            "params": {
                "n_clusters": [100, 200]
            }
        }
    }
    
    opt_a = create_optimizer(config_a)
    opt_b = create_optimizer(config_b)
    
    comb_a = opt_a._generate_hyperparameter_combinations()
    comb_b = opt_b._generate_hyperparameter_combinations()
    
    # Compare only the varied_params part for simplicity
    params_a = [c[1] for c in comb_a]
    params_b = [c[1] for c in comb_b]
    
    assert params_a == params_b
