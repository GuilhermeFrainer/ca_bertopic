import polars as pl
import numpy as np

from typing import Union, Optional
import logging


def load_and_prep_data(config: dict, random_state: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Loads parquet, samples data, and processes metadata.
    """
    logger = logging.getLogger("pipeline")
    experiment_config: dict = config["experiment"]

    data_path = experiment_config["dataset_path"]
    sample_size = experiment_config.get("sample_size")

    covariates_config = experiment_config["covariates"]

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
    
    # We must drop empty rows, as we can't compute Coherence scores for them
    non_empty_lf = lf.filter(pl.col(text_col) != "")

    # Identify required columns for selection
    if isinstance(covariates_config, list):
        cov_cols = covariates_config
    else:
        cov_cols = (
            covariates_config.get("numerical", []) +
            covariates_config.get("categorical", []) +
            covariates_config.get("binary", [])
        )
    
    # Deduplicate required columns
    relevant_cols = list(set([text_col, embedding_col] + cov_cols))

    try:
        df = non_empty_lf.select(relevant_cols).collect()
    except pl.exceptions.ColumnNotFoundError:
        available_cols = full_lf.collect_schema().names()
        logger.error(f"Column not found in dataset. Available columns: {available_cols}")
        raise

    dropped_rows = dataset_len - len(df)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows for being empty strings.")
        logger.info(f"Running experiment on {len(df)} rows.")
    
    text = df[text_col].to_list()
    embeddings = df[embedding_col].to_numpy()
    
    processed_metadata = process_metadata(df, covariates_config)

    return text, embeddings, processed_metadata


def process_metadata(df: pl.DataFrame, covariates_config: Union[dict, list]) -> np.ndarray:
    """
    Parses the covariates config and applies specific scaling/encoding 
    strategies for Numerical, Categorical, and Binary variables.
    """
    logger = logging.getLogger("pipeline")

    # Parse configuration
    # Legacy config
    if isinstance(covariates_config, list):
        logger.warning("Deprecation Warning: 'covariates' is a list. Assuming all are numerical.")
        num_cols = covariates_config
        cat_cols = []
        bin_cols = []
    else:
        num_cols = covariates_config.get("numerical", [])
        cat_cols = covariates_config.get("categorical", [])
        bin_cols = covariates_config.get("binary", [])

    processed_features = []

    # Min-max scaling for numerical columns
    if num_cols:
        logger.info(f"Processing Numerical cols: {num_cols}")
        # Check if columns exist
        missing = [c for c in num_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing numerical columns: {missing}")

        num_df = df.select(num_cols)
        # Apply MinMax Scaling safely
        exprs = []
        for c in num_cols:
            c_min = pl.col(c).min()
            c_max = pl.col(c).max()
            # Avoid division by zero if max == min
            exprs.append(
                pl.when(c_max != c_min)
                .then((pl.col(c) - c_min) / (c_max - c_min))
                .otherwise(0.0) 
            )
            
        scaled_num = num_df.with_columns(exprs).fill_nan(0.0).to_numpy()
        processed_features.append(scaled_num)

    # One-hot encoding for categorical values
    if cat_cols:
        logger.info(f"Processing Categorical cols: {cat_cols}")
        missing = [c for c in cat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing categorical columns: {missing}")
            
        dummies_df = df.select(cat_cols).to_dummies(drop_first=False)
        processed_features.append(dummies_df.to_numpy())

    # Binary variables are simply cast to float
    if bin_cols:
        logger.info(f"Processing Binary cols: {bin_cols}")
        missing = [c for c in bin_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing binary columns: {missing}")
            
        bin_matrix = df.select(bin_cols).select(pl.all().cast(pl.Float64)).to_numpy()
        processed_features.append(bin_matrix)

    if processed_features:
        final_metadata = np.hstack(processed_features)
        logger.info(f"Metadata processing complete. Shape: {final_metadata.shape}")
        return final_metadata
    else:
        logger.warning("No covariates found in config. Returning empty array.")
        return np.array([])


def sample_from_lf(
    lf: pl.LazyFrame,
    n: int,
    seed: Optional[int] = None,
    replace: bool = False
) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)

    lf_len = lf.select(pl.len()).collect().item()
    if n > lf_len and not replace:
        raise ValueError(f"Cannot sample {n} rows without replacement from a dataset of {lf_len} rows.")

    # Determines id column
    schema = lf.collect_schema()
    cols = schema.names()
    if "index" in cols:
        id_col = "index"
    elif "id" in cols:
        id_col = "id"
    else:
        raise ValueError((
            "Unable to find ID column to sample from LazyFrame"
            "Available columns:\n"
            f"{cols}"
        ))

    # numpy's choice will raise a ValueError if sampling without replacement is
    # not possible, which is the desired behavior.
    sample_idxs = rng.choice(lf_len, size=n, replace=replace)

    # Create a LazyFrame of sampled indices.
    sampled_indices_lf = pl.DataFrame({id_col: sample_idxs}).lazy()

    # We must join the sampled indices back to the original LazyFrame.
    # Using `filter(pl.col("index").is_in(sample_idxs))` would not work for
    # sampling with replacement, as `is_in` only filters unique values.
    # A join correctly preserves all sampled rows, including duplicates.
    return sampled_indices_lf.join(lf, on=id_col, how="inner")

