from pathlib import Path

import numpy as np
import pytest

from src.models import create_bertopic_instance
from src.optimizer import generate_hyperparameter_combinations
from src.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Collect all standard experiment config paths (excluding archive)
STANDARD_CONFIG_FILES = [
    p for p in EXPERIMENTS_DIR.glob("*/*_standard_*.yaml") if "archive" not in p.parts
]


def test_standard_configs_exist():
    """Verify that standard experiment configuration files exist."""
    assert len(STANDARD_CONFIG_FILES) >= 50, (
        f"Expected at least 50 standard configs, found {len(STANDARD_CONFIG_FILES)}"
    )


@pytest.mark.parametrize("config_path", STANDARD_CONFIG_FILES, ids=lambda p: p.stem)
def test_standard_config_instantiation(config_path: Path):
    """
    Validates that each standard experiment config:
    1. Loads properly with inheritance.
    2. Instantiates BERTopic models via optimizer or direct config loading.
    """
    rel_path = config_path.relative_to(EXPERIMENTS_DIR)
    config = load_config(str(rel_path), EXPERIMENTS_DIR)

    assert "experiment" in config

    # Skip STM configs as they are handled by R scripts and specify 'models' list
    if "stm" in config_path.stem:
        assert "models" in config
        return

    assert "model" in config
    model_config = config["model"]

    # If parameters are present for tuning/expanding
    combinations = generate_hyperparameter_combinations(model_config)

    # Mock metadata (e.g. 20 samples, 4 features)
    scaled_metadata = np.random.rand(20, 4)

    for combo_config, _ in combinations:
        topic_model = create_bertopic_instance(
            model_config=combo_config,
            scaled_metadata=scaled_metadata,
            random_state=36201624,
        )
        assert topic_model is not None
