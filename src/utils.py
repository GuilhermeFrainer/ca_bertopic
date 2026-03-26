import logging
from pathlib import Path
import yaml
import random

import polars as pl


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


def extract_qualitative_data(topic_model, model_id: str, metadata: dict) -> pl.DataFrame:
    """
    Extracts c-TF-IDF words and representative documents from a BERTopic model.

    Args:
        topic_model: Fitted BERTopic model.
        model_id: Identifier for the model.
        metadata: Dictionary of metadata/hyperparameters to include.

    Returns:
        Polars DataFrame with topic information and metadata.
    """
    import polars as pl
    import json

    # Get topic info from BERTopic (returns a pandas DataFrame)
    topic_info = topic_model.get_topic_info()

    # Convert to Polars
    df = pl.from_pandas(topic_info)

    # Standardize column names to snake_case
    rename_dict = {
        "Topic": "topic_id",
        "Count": "count",
        "Name": "name",
        "Representation": "representation",
        "Representative_Docs": "representative_docs"
    }
    # Only rename if they exist
    actual_rename = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(actual_rename)

    # Add model_id at the front
    df = df.with_columns(pl.lit(model_id).alias("model_id"))

    # Add metadata columns at the back
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        df = df.with_columns(pl.lit(value).alias(key))

    # Reorder columns: model_id first, then topic info, then metadata
    topic_cols = ["topic_id", "count", "name", "representation", "representative_docs"]
    existing_topic_cols = [c for c in topic_cols if c in df.columns]
    metadata_keys = list(metadata.keys())

    final_order = ["model_id"] + existing_topic_cols + metadata_keys
    return df.select(final_order)
        
