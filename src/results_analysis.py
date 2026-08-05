from typing import Dict, List

import polars as pl

METRICS = ["u_mass", "c_v", "c_npmi", "irbo", "topic_diversity"]


def extract_model_type(model_name: str, merge_info0: bool = False) -> str:
    """Extracts the base model type from a model name.

    Example: 'baseline_1' -> 'baseline', 'baseline_seed36201624' -> 'baseline'.
    """
    if not isinstance(model_name, str):
        return str(model_name)

    res = model_name
    # Strip common prefixes
    if res.startswith("stemmed_"):
        res = res[len("stemmed_") :]

    if res.startswith("stm_"):
        return "stm"

    # Remove seed suffixes if present (e.g., _seed36201624)
    if "_seed" in res:
        res = res.split("_seed")[0]

    # Remove any trailing numbers (e.g., baseline_1 -> baseline)
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
    Finds the best performing model (or average performance with standard deviation)
    of each model type for each metric in the given dataset.

    Args:
        df: Polars DataFrame containing experiment results.
        dataset: The name of the dataset to filter by.
        exclude_clustering: Optional list of clustering algorithms to exclude.
        exclude_dim_red: Optional list of dimensionality reduction
            algorithms to exclude.
        dump: If True, returns all model configurations instead of
            just the best per type.
        average: If True, calculates the average performance and standard deviation
            per model type across runs/seeds.
        merge_info0: If True, treats models with and without _info0 as the same type.
        suppress_nulls: If True, filters out rows with NaNs in any metric column.

    Returns:
        A dictionary where keys are metric names and values are DataFrames
        with models and their scores (including max_value, std_value, n_seeds).
    """
    # Normalize dataset name and model names
    if "dataset_name" in df.columns:
        df = df.with_columns(
            pl.col("dataset_name")
            .replace("anes_stemmed", "anes")
            .str.replace(r"_s\d+$", "")
        )

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
                    pl.lit(0.0).alias("std_value"),
                    pl.lit(1).alias("n_seeds"),
                    pl.col("model_type"),
                ]
            )
        elif average:
            # For average mode, group by model_type and calculate mean, std, n_seeds
            best_per_type = (
                metric_df.group_by("model_type")
                .agg(
                    pl.col(metric).mean().alias("max_value"),
                    pl.col(metric).std().fill_null(0.0).alias("std_value"),
                    pl.col(metric).count().alias("n_seeds"),
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
                    pl.lit(0.0).alias("std_value"),
                    pl.lit(1).alias("n_seeds"),
                    pl.col("model_name").first().alias("best_model_name"),
                )
                .sort("max_value", descending=True)
            )
        results[metric] = best_per_type

    return results


def calculate_hdbscan_noise_coverage(
    df: pl.DataFrame,
    dataset: str | None = None,
    group_by_model_type: bool = True,
    merge_info0: bool = False,
) -> pl.DataFrame:
    """Calculates noise-cluster coverage for models using HDBSCAN clustering.

    Noise-cluster coverage measures the proportion/percentage of documents assigned
    to the noise cluster (outlier topic -1) relative to total document observations.

    Args:
        df: Polars DataFrame containing experiment results.
        dataset: Optional dataset name to filter by.
        group_by_model_type: If True, aggregates across seeds per model type.
        merge_info0: If True, treats models with and without _info0 as same type.

    Returns:
        Polars DataFrame containing noise coverage metrics.
    """
    if df.is_empty():
        return pl.DataFrame()

    # Filter for HDBSCAN clustering algorithm
    if "clustering_algo" in df.columns:
        filtered_df = df.filter(pl.col("clustering_algo") == "hdbscan")
    else:
        filtered_df = df

    if dataset and "dataset_name" in filtered_df.columns:
        filtered_df = filtered_df.filter(pl.col("dataset_name") == dataset)

    if filtered_df.is_empty():
        return pl.DataFrame()

    # Ensure required columns exist
    if (
        "outliers" not in filtered_df.columns
        or "n_observations" not in filtered_df.columns
    ):
        raise ValueError(
            "DataFrame must contain 'outliers' and 'n_observations' columns."
        )

    # Calculate individual noise coverage ratio and percentage
    filtered_df = filtered_df.with_columns(
        (pl.col("outliers") / pl.col("n_observations")).alias("noise_ratio"),
        ((pl.col("outliers") / pl.col("n_observations")) * 100.0).alias(
            "noise_coverage_pct"
        ),
        ((1.0 - (pl.col("outliers") / pl.col("n_observations"))) * 100.0).alias(
            "clustered_coverage_pct"
        ),
    )

    if not group_by_model_type:
        return filtered_df

    # Extract model_type if needed
    if "model_type" not in filtered_df.columns:
        filtered_df = filtered_df.with_columns(
            pl.col("model_name")
            .map_elements(
                lambda x: extract_model_type(x, merge_info0=merge_info0),
                return_dtype=pl.String,
            )
            .alias("model_type")
        )

    group_cols = ["model_type"]
    if "dataset_name" in filtered_df.columns:
        group_cols.insert(0, "dataset_name")

    aggregated = (
        filtered_df.group_by(group_cols)
        .agg(
            pl.len().alias("n_runs"),
            pl.col("n_observations").first().alias("n_observations"),
            pl.col("outliers").mean().round(2).alias("outliers_mean"),
            pl.col("outliers").std().fill_null(0.0).round(2).alias("outliers_std"),
            pl.col("noise_coverage_pct")
            .mean()
            .round(2)
            .alias("noise_coverage_pct_mean"),
            pl.col("noise_coverage_pct")
            .std()
            .fill_null(0.0)
            .round(2)
            .alias("noise_coverage_pct_std"),
            pl.col("noise_coverage_pct").min().round(2).alias("noise_coverage_pct_min"),
            pl.col("noise_coverage_pct").max().round(2).alias("noise_coverage_pct_max"),
            pl.col("clustered_coverage_pct")
            .mean()
            .round(2)
            .alias("clustered_coverage_pct_mean"),
        )
        .sort(group_cols)
    )

    return aggregated
