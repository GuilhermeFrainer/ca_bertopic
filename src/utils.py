import logging
from pathlib import Path
import yaml
import random


def load_config(exp_name: str, experiments_dir: Path) -> dict:
    """
    Resolves path and loads the YAML config
    """
    logger = logging.getLogger("pipeline")

    filename = exp_name if exp_name.endswith(".yaml") else f"{exp_name}.yaml"
    config_path = experiments_dir / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file {config_path} not found.")

    with open(config_path, "r") as f:
        logger.info(f"Loaded config from {config_path}")
        return yaml.safe_load(f)


def get_random_state(random_state: str | int) -> int:
    if isinstance(random_state, int):
        return random_state
    elif random_state == "random":
        return random.randint(0, 100_000)
    else:
        raise ValueError(f"Invalid random state: {random_state}")
        
