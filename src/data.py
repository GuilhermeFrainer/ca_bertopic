import polars as pl
import numpy as np

from typing import Optional
import logging


def load_and_prep_data(config: dict, random_state: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Loads parquet, samples data, and scales metadata.
    """
    def min_max_scaler(col: str):
        x = pl.col(col)
        return (x - x.min()) / (x.max() - x.min())
    
    logger = logging.getLogger("pipeline")

    experiment_config: dict = config["experiment"]

    data_path = experiment_config["dataset_path"]
    sample_size = experiment_config.get("sample_size")
    covariates = experiment_config["covariates"]

    text_col = experiment_config.get("text_col", "text")
    embedding_col = experiment_config.get("embedding_col", "embedding")

    logger.info(f"Target Text Column: '{text_col}'")
    logger.info(f"Target Embedding Column: '{embedding_col}'")

    # Lazy load and sample
    full_lf = pl.scan_parquet(data_path)
    dataset_len = full_lf.select(text_col).count().collect().item()

    if sample_size is not None:
        logger.info(f"Subsampling dataset to {sample_size} rows.")
        lf = sample_from_lf(full_lf, n=sample_size, seed=random_state)
    else:
        logger.info(f"Using full dataset. Total rows: {dataset_len}")
        lf = full_lf
    
    # Materialize data
    # We must drop empty rows, as we can't compute Coherence scores for them
    non_empty_lf = lf.filter(pl.col(text_col) != "")
    df = non_empty_lf.collect()

    dropped_rows = dataset_len - len(df)
    logger.info(f"Dropped {dropped_rows} rows for being empty strings.")
    
    try:
        text = df[text_col].to_list()
        embeddings = df[embedding_col].to_numpy()
    except pl.exceptions.ColumnNotFoundError as e:
        logger.error(f"Column not found in dataset. Available columns: {df.columns}")
        raise e
    
    # Metadata scaling
    metadata_df = df.select(covariates)
    scaling_expressions = [min_max_scaler(c) for c in metadata_df.columns]
    scaled_metadata = metadata_df.with_columns(scaling_expressions).to_numpy()

    return text, embeddings, scaled_metadata


def sample_from_lf(
    lf: pl.LazyFrame,
    n: int,
    seed: Optional[int] = None,
    replace: bool = False
) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    lf_len = lf.select("index").count().collect().item()
    all_possible_rows = np.arange(lf_len)
    sample_idxs = rng.choice(all_possible_rows, size=n, replace=replace)
    return lf.filter(pl.col("index").is_in(sample_idxs))

