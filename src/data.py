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

    data_path = config["experiment"]["dataset_path"]
    sample_size = config["experiment"].get("sample_size")
    covariates = config["experiment"]["covariates"]

    # Lazy load and sample
    full_lf = pl.scan_parquet(data_path)

    if sample_size is not None:
        logger.info(f"Subsampling dataset to {sample_size} rows.")
        lf = sample_from_lf(full_lf, n=sample_size, seed=random_state)
    else:
        dataset_len = full_lf.select("text").count().collect().item()
        logger.info(f"Using full dataset. Total rows: {dataset_len}")
        lf = full_lf
    
    # Materialize data
    df = lf.collect()
    
    text = df["text"].to_list()
    try:
        embeddings = df["embedding"].to_numpy()
    except pl.exceptions.ColumnNotFoundError:
        embeddings = df["text_embedding"].to_numpy()
    
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

