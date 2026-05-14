from typing import Dict, List

import polars as pl

METRICS = ["u_mass", "c_v", "c_npmi", "irbo", "topic_diversity"]


def extract_model_type(model_name: str, merge_info0: bool = False) -> str:
    """Extracts the base model type from a model name.

    Example: 'baseline_1' -> 'baseline'.
    """
    if not isinstance(model_name, str):
        return str(model_name)

    res = model_name
    # Strip common prefixes
    if res.startswith("stemmed_"):
        res = res[len("stemmed_") :]

    # First remove any trailing numbers (e.g., baseline_1 -> baseline)
    if "_" in res:
        parts = res.split("_")
        if parts[-1].isdigit():
            res = "_".join(parts[:-1])

    # Then optionally strip info0
    if merge_info0 and res.endswith("_info0"):
        res = res[: -len("_info0")]

    return res


def find_best_models(
    df: pl.DataFrame,
    dataset: str,
    exclude_clustering: List[str] | None = None,
    exclude_dim_red: List[str] | None = None,
    dump: bool = False,
    average: bool = False,
    merge_info0: bool = False,
    suppress_nulls: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    Finds the best performing model (or average performance) of each model type
    for each metric in the given dataset.

    Args:
        df: Polars DataFrame containing experiment results.
        dataset: The name of the dataset to filter by.
        exclude_clustering: Optional list of clustering algorithms to exclude.
        exclude_dim_red: Optional list of dimensionality reduction
            algorithms to exclude.
        dump: If True, returns all model configurations instead of
            just the best per type.
        average: If True, calculates the average performance per model type
            instead of finding the best.
        merge_info0: If True, treats models with and without _info0 as the same type.
        suppress_nulls: If True, filters out rows with NaNs in any metric column.

    Returns:
        A dictionary where keys are metric names and values are DataFrames
        with models and their scores.
    """
    # Normalize dataset name and model names
    if "dataset_name" in df.columns:
        df = df.with_columns(pl.col("dataset_name").replace("anes_stemmed", "anes"))

    if "model_name" in df.columns:
        df = df.with_columns(pl.col("model_name").str.replace("^stemmed_", ""))

    # Filter by dataset
    if "dataset_name" in df.columns:
        df = df.filter(pl.col("dataset_name") == dataset)

    if df.is_empty():
        return {}

    # Optional: Suppress rows with any NaNs in metric columns
    if suppress_nulls:
        actual_metric_cols = [m for m in METRICS if m in df.columns]
        if actual_metric_cols:
            # Drop rows where any metric column has a NaN/Null
            # We cast to float to ensure uniform check
            null_mask = pl.any_horizontal(
                pl.col(actual_metric_cols).cast(pl.Float64, strict=False).is_null()
                | pl.col(actual_metric_cols).cast(pl.Float64, strict=False).is_nan()
            )
            df = df.filter(~null_mask)

    if df.is_empty():
        return {}

    # Apply exclusion filters
    # Default: exclude PCA dim red and any k_means variant if not explicitly provided
    if "clustering_algo" in df.columns:
        if exclude_clustering is not None:
            df = df.filter(~pl.col("clustering_algo").is_in(exclude_clustering))
        else:
            df = df.filter(~pl.col("clustering_algo").str.contains("k_means"))

    if "dim_red_algo" in df.columns:
        if exclude_dim_red is not None:
            df = df.filter(~pl.col("dim_red_algo").is_in(exclude_dim_red))
        else:
            df = df.filter(pl.col("dim_red_algo") != "pca")

    if df.is_empty():
        return {}

    # Add model_type column if it doesn't exist
    if "model_type" not in df.columns:
        df = df.with_columns(
            pl.col("model_name")
            .map_elements(
                lambda x: extract_model_type(x, merge_info0=merge_info0),
                return_dtype=pl.String,
            )
            .alias("model_type")
        )

    available_metrics = [m for m in METRICS if m in df.columns]

    results = {}
    for metric in available_metrics:
        # Ensure metric is numeric and filter out nulls/NaNs
        if df[metric].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            continue

        # filter out nulls and NaNs
        metric_df = df.filter(pl.col(metric).is_not_null())
        if metric_df[metric].dtype in [pl.Float32, pl.Float64]:
            metric_df = metric_df.filter(~pl.col(metric).is_nan())

        if metric_df.is_empty():
            continue

        if dump:
            # For dump mode, just sort by the metric and keep everything
            best_per_type = metric_df.sort(metric, descending=True).select(
                [
                    pl.col("model_name").alias("best_model_name"),
                    pl.col(metric).alias("max_value"),
                    pl.col("model_type"),
                ]
            )
        elif average:
            # For average mode, group by model_type and calculate mean
            best_per_type = (
                metric_df.group_by("model_type")
                .agg(
                    pl.col(metric).mean().alias("max_value"),
                    pl.col("model_type").first().alias("best_model_name"),
                )
                .sort("max_value", descending=True)
            )
        else:
            # For each metric, group by model_type and find the max
            best_per_type = (
                metric_df.sort(metric, descending=True)
                .group_by("model_type")
                .agg(
                    pl.col(metric).first().alias("max_value"),
                    pl.col("model_name").first().alias("best_model_name"),
                )
                .sort("max_value", descending=True)
            )
        results[metric] = best_per_type

    return results
