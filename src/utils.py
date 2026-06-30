import logging
import random
from pathlib import Path

import numpy as np
import polars as pl
import yaml


def extract_stm_qualitative_data(
    theta: np.ndarray,
    beta: np.ndarray,
    vocab: list[str],
    documents: list[str],
    model_id: str,
    metadata: dict,
    topk: int = 10,
) -> pl.DataFrame:
    """
    Extracts qualitative data from STM outputs (theta, beta) in a format
    compatible with the existing BERTopic qualitative data schema.
    """
    import json

    n_topics = beta.shape[0]

    # 1. Topic counts (sum of probabilities)
    counts = np.sum(theta, axis=0)

    # 2. Representations (top words)
    top_words = []
    for i in range(n_topics):
        top_indices = np.argsort(beta[i])[::-1][:topk]
        top_words.append([vocab[idx] for idx in top_indices])

    # 3. Representative documents (top 3 documents for each topic)
    rep_docs = []
    for i in range(n_topics):
        top_doc_indices = np.argsort(theta[:, i])[::-1][:3]
        # Ensure we don't go out of bounds if documents is shorter (shouldn't happen)
        actual_indices = [idx for idx in top_doc_indices if idx < len(documents)]
        rep_docs.append([documents[idx] for idx in actual_indices])

    # 4. Create DataFrame
    data = []
    for i in range(n_topics):
        topic_id = i
        data.append(
            {
                "topic_id": topic_id,
                "count": int(counts[i]),
                "name": f"{topic_id}_" + "_".join(top_words[i][:3]),
                "representation": top_words[i],
                "representative_docs": rep_docs[i],
            }
        )

    df = pl.DataFrame(data)

    # Add model_id at the front
    df = df.with_columns(pl.lit(model_id).alias("model_id"))

    # Add metadata columns at the back
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        df = df.with_columns(pl.lit(value).alias(key))

    # Reorder columns
    topic_cols = ["topic_id", "count", "name", "representation", "representative_docs"]
    metadata_keys = list(metadata.keys())
    final_order = ["model_id"] + topic_cols + metadata_keys

    return df.select(final_order)


def load_config(exp_name: str, experiments_dir: Path) -> dict:
    """
    Resolves path and loads the YAML config, supporting inheritance via 'extends'.
    """
    logger = logging.getLogger("pipeline")

    filename = exp_name if exp_name.endswith(".yaml") else f"{exp_name}.yaml"
    config_path = experiments_dir / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file {config_path} not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "extends" in config:
        base_rel_path = config.pop("extends")
        # Ensure we can load from subdirectories like 'datasets/'
        base_path = experiments_dir / base_rel_path

        if not base_path.exists():
            raise FileNotFoundError(f"Base config file {base_path} not found.")

        with open(base_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
            logger.info(f"Loaded base config from {base_path}")

        # If base file is flat (no 'experiment' key), treat it as 'experiment' data
        if (
            "experiment" not in base_config
            and "models" not in base_config
            and "model" not in base_config
        ):
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


def get_random_state(random_state: str | int | list[str | int]) -> int | list[int]:
    if isinstance(random_state, list):
        return [get_random_state(r) for r in random_state]
    if isinstance(random_state, int):
        return random_state
    elif random_state == "random":
        return random.randint(0, 100_000)
    else:
        raise ValueError(f"Invalid random state: {random_state}")


def extract_qualitative_data(
    topic_model, model_id: str, metadata: dict
) -> pl.DataFrame:
    """
    Extracts c-TF-IDF words and representative documents from a BERTopic model.

    Args:
        topic_model: Fitted BERTopic model.
        model_id: Identifier for the model.
        metadata: Dictionary of metadata/hyperparameters to include.

    Returns:
        Polars DataFrame with topic information and metadata.
    """
    import json

    import polars as pl

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
        "Representative_Docs": "representative_docs",
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
