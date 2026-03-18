import logging
from pathlib import Path
import yaml
import random


def load_config(exp_name: str, experiments_dir: Path) -> dict:
    """
    Resolves path and loads the YAML config, supporting inheritance via 'extends'.
    """
    logger = logging.getLogger("pipeline")

    filename = exp_name if exp_name.endswith(".yaml") else f"{exp_name}.yaml"
    config_path = experiments_dir / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file {config_path} not found.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "extends" in config:
        base_rel_path = config.pop("extends")
        # Ensure we can load from subdirectories like 'datasets/'
        base_path = experiments_dir / base_rel_path
        
        if not base_path.exists():
            raise FileNotFoundError(f"Base config file {base_path} not found.")

        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)
            logger.info(f"Loaded base config from {base_path}")

        # If base file is flat (no 'experiment' key), treat it as 'experiment' data
        if "experiment" not in base_config and "models" not in base_config and "model" not in base_config:
            base_config = {"experiment": base_config}

        # Merge logic (Replace strategy)
        for key, value in config.items():
            if key == "experiment" and isinstance(value, dict) and key in base_config:
                # Merge the 'experiment' dictionary keys (shallow merge)
                base_config[key].update(value)
            else:
                # Replace other top-level keys (e.g., 'models', 'model')
                base_config[key] = value
        
        config = base_config

    logger.info(f"Loaded config from {config_path}")
    return config


def get_random_state(random_state: str | int) -> int:
    if isinstance(random_state, int):
        return random_state
    elif random_state == "random":
        return random.randint(0, 100_000)
    else:
        raise ValueError(f"Invalid random state: {random_state}")
        
