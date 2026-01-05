import logging
import datetime
from pathlib import Path
import yaml


def setup_logging(experiment_name: str, log_dir: Path) -> logging.Logger:
    """
    Sets up a logger that writes to file and console.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"{experiment_name}-{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Keeps printing to console
        ]
    )
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.info(f"Starting experiment: {experiment_name}")
    return logger


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

