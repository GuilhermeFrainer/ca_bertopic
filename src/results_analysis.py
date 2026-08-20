from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import scipy.stats as scipy_stats

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


def compute_stopword_impact(
    df_remove_rep_stopwords: pl.DataFrame,
    df_keep_rep_stopwords: pl.DataFrame,
    dataset: str = "fed",
    exclude_clustering: List[str] | None = None,
    exclude_dim_red: List[str] | None = None,
    merge_info0: bool = False,
) -> Dict[str, pl.DataFrame]:
    """Computes metric differences between remove and keep stopwords runs.

    Calculates pairwise difference (remove_rep_stopwords - keep_rep_stopwords)
    for corresponding models (same configuration and seed) and aggregates
    by model_type.

    Args:
        df_remove_rep_stopwords: Polars DataFrame of results with representation
            stopwords removed.
        df_keep_rep_stopwords: Polars DataFrame of results with representation
            stopwords kept.
        dataset: Name of dataset to filter by.
        exclude_clustering: Optional list of clustering algorithms to exclude.
        exclude_dim_red: Optional list of dim reduction algorithms to exclude.
        merge_info0: If True, treats _info0 model variants as their base model type.

    Returns:
        A dictionary mapping each metric to a Polars DataFrame with columns:
        ["model_type", "mean_delta", "std_delta", "n_pairs"].
    """
    df_standard = df_remove_rep_stopwords
    df_no_stopword = df_keep_rep_stopwords

    if df_standard.is_empty() or df_no_stopword.is_empty():
        return {}

    # Standardize dataset column if present
    for df in [df_standard, df_no_stopword]:
        if "dataset_name" in df.columns:
            df = df.with_columns(
                pl.col("dataset_name")
                .replace("anes_stemmed", "anes")
                .str.replace(r"_s\d+$", "")
            )

    if "dataset_name" in df_standard.columns:
        df_standard = df_standard.filter(pl.col("dataset_name") == dataset)
    if "dataset_name" in df_no_stopword.columns:
        df_no_stopword = df_no_stopword.filter(pl.col("dataset_name") == dataset)

    if df_standard.is_empty() or df_no_stopword.is_empty():
        return {}

    # Strip stemmed_ prefix
    if "model_name" in df_standard.columns:
        df_standard = df_standard.with_columns(
            pl.col("model_name").str.replace("^stemmed_", "")
        )
    if "model_name" in df_no_stopword.columns:
        df_no_stopword = df_no_stopword.with_columns(
            pl.col("model_name").str.replace("^stemmed_", "")
        )

    # Filter clustering and dim reduction algorithms
    if "clustering_algo" in df_standard.columns:
        if exclude_clustering is not None:
            df_standard = df_standard.filter(
                ~pl.col("clustering_algo").is_in(exclude_clustering)
            )
        else:
            df_standard = df_standard.filter(
                ~pl.col("clustering_algo").str.contains("k_means")
            )
    if "clustering_algo" in df_no_stopword.columns:
        if exclude_clustering is not None:
            df_no_stopword = df_no_stopword.filter(
                ~pl.col("clustering_algo").is_in(exclude_clustering)
            )
        else:
            df_no_stopword = df_no_stopword.filter(
                ~pl.col("clustering_algo").str.contains("k_means")
            )

    if "dim_red_algo" in df_standard.columns:
        if exclude_dim_red is not None:
            df_standard = df_standard.filter(
                ~pl.col("dim_red_algo").is_in(exclude_dim_red)
            )
        else:
            df_standard = df_standard.filter(pl.col("dim_red_algo") != "pca")
    if "dim_red_algo" in df_no_stopword.columns:
        if exclude_dim_red is not None:
            df_no_stopword = df_no_stopword.filter(
                ~pl.col("dim_red_algo").is_in(exclude_dim_red)
            )
        else:
            df_no_stopword = df_no_stopword.filter(pl.col("dim_red_algo") != "pca")

    if df_standard.is_empty() or df_no_stopword.is_empty():
        return {}

    # Join standard and no_stopword dataframes
    if "model_name" in df_standard.columns and "model_name" in df_no_stopword.columns:
        # Deduplicate before join to prevent cartesian products
        std_unique = df_standard.unique(subset=["model_name"])
        no_unique = df_no_stopword.unique(subset=["model_name"])
        joined = std_unique.join(no_unique, on="model_name", suffix="_no_stopword")
    else:
        # Fallback to join on random_state and n_topics if available
        join_cols = [
            c
            for c in ["random_state", "n_topics", "clustering_algo", "dim_red_algo"]
            if c in df_standard.columns and c in df_no_stopword.columns
        ]
        if not join_cols:
            return {}
        joined = df_standard.join(df_no_stopword, on=join_cols, suffix="_no_stopword")

    if joined.is_empty():
        return {}

    # Map model_type
    joined = joined.with_columns(
        pl.col("model_name")
        .map_elements(
            lambda x: extract_model_type(x, merge_info0=merge_info0),
            return_dtype=pl.String,
        )
        .alias("model_type")
    )

    available_metrics = [m for m in METRICS if m in joined.columns]
    results = {}

    for metric in available_metrics:
        ns_col = f"{metric}_no_stopword"
        if ns_col not in joined.columns:
            continue

        metric_df = joined.filter(
            pl.col(metric).is_not_null() & pl.col(ns_col).is_not_null()
        )
        if metric_df[metric].dtype in [pl.Float32, pl.Float64]:
            metric_df = metric_df.filter(
                ~pl.col(metric).is_nan() & ~pl.col(ns_col).is_nan()
            )

        if metric_df.is_empty():
            continue

        metric_df = metric_df.with_columns(
            (pl.col(metric) - pl.col(ns_col)).alias("delta")
        )

        aggregated = (
            metric_df.group_by("model_type")
            .agg(
                pl.col("delta").mean().alias("mean_delta"),
                pl.col("delta").std().fill_null(0.0).alias("std_delta"),
                pl.col("delta").count().alias("n_pairs"),
                pl.col("model_type").first().alias("best_model_name"),
            )
            .sort("mean_delta", descending=True)
        )
        results[metric] = aggregated

    return results


def parse_model_type_and_topic(
    model_name: str, merge_info0: bool = False
) -> Tuple[str, int]:
    """Extracts base model type and topic count index from a model run name.

    Examples:
        'baseline_1_seed36201624' -> ('baseline', 1)
        'stemmed_mv_spectral_3_seed100' -> ('mv_spectral', 3)
        'append_umap_5' -> ('append_umap', 5)

    Args:
        model_name: The raw experiment model name.
        merge_info0: Whether to strip trailing '_info0'.

    Returns:
        Tuple of (model_type, topic_index).
    """
    if not isinstance(model_name, str):
        return str(model_name), 0

    clean = model_name
    if clean.startswith("stemmed_"):
        clean = clean[len("stemmed_") :]
    if clean.startswith("stm_"):
        # Handle stm naming if needed
        pass
    if "_seed" in clean:
        clean = clean.split("_seed")[0]

    parts = clean.split("_")
    if parts[-1].isdigit():
        topic_idx = int(parts[-1])
        base = "_".join(parts[:-1])
    else:
        topic_idx = 0
        base = clean

    if merge_info0 and base.endswith("_info0"):
        base = base[: -len("_info0")]

    return base, topic_idx


def wilcoxon_exact_test(
    differences: Sequence[float] | np.ndarray,
    alternative: str = "two-sided",
    zero_method: str = "pratt",
) -> Tuple[float, float]:
    """Computes exact paired Wilcoxon signed-rank test statistic and p-value.

    Following Demšar (2006), uses the exact permutation distribution for small
    sample sizes (such as N=5 topic counts) and applies Pratt's method for handling
    zero differences without discarding paired blocks.

    Args:
        differences: Paired differences (Alternative - Default) across sample blocks.
        alternative: Test direction ('two-sided', 'greater', or 'less').
        zero_method: Zero handling method ('pratt', 'wilcoxon', or 'zsplit').

    Returns:
        Tuple of (statistic, p_value).
    """
    diff_arr = np.asarray(differences, dtype=float)
    if len(diff_arr) == 0:
        return 0.0, 1.0

    # If all differences are zero, there is no variance and no difference
    if np.all(diff_arr == 0):
        return 0.0, 1.0

    try:
        res = scipy_stats.wilcoxon(
            diff_arr,
            zero_method=zero_method,
            method="exact",
            alternative=alternative,
        )
        p_val = float(res.pvalue)
        if np.isnan(p_val):
            p_val = 1.0
        return float(res.statistic), p_val
    except Exception:
        # Fallback for degenerate distributions
        return 0.0, 1.0


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down adjustment to control FWER.

    Controls family-wise error rate across a family of hypotheses:
    p_adj_(i) = min(1.0, max_{j <= i} ((m - j + 1) * p_(j)))

    Args:
        p_values: Sequence of raw p-values.

    Returns:
        List of adjusted p-values in original input order.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        multiplier = n - rank
        adj = min(1.0, multiplier * p)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max

    return adjusted


def compute_demsar_delta_table(
    df_default: pl.DataFrame,
    df_alternative: pl.DataFrame,
    dataset: Optional[str] = "fed",
    metrics: Optional[List[str]] = None,
    alpha: float = 0.10,
    alternative: str = "two-sided",
    zero_method: str = "pratt",
    correction: str = "per_metric",
    exclude_clustering: Optional[List[str]] = None,
    exclude_dim_red: Optional[List[str]] = None,
    merge_info0: bool = False,
) -> Dict[str, Any]:
    """Computes Demšar-compliant Model-by-Metric Delta Table and statistical tests.

    Workflow:
    1. Seed Aggregation: Averages scores across random seeds for each
       (model, metric, topic_count).
    2. Delta Computation: Computes paired differences across N=5 topic counts.
    3. Non-Parametric Significance: Runs paired exact Wilcoxon signed-rank test
       for each (model, metric).
    4. FWER Control: Applies Holm-Bonferroni step-down correction.
    5. Formatting: Creates structured delta tables with significance flags.

    Args:
        df_default: Polars DataFrame of baseline/default runs.
        df_alternative: Polars DataFrame of alternative configuration runs.
        dataset: Dataset identifier to filter by (or None for all).
        metrics: List of metrics to evaluate (default: METRICS).
        alpha: Significance threshold (e.g., 0.10 for two-tailed or 0.05
            for one-tailed).
        alternative: Test direction ('two-sided', 'greater', or 'less').
        zero_method: Zero handling method for Wilcoxon ('pratt' or 'wilcoxon').
        correction: FWER correction mode ('per_metric', 'table', or 'none').
        exclude_clustering: Clustering algorithms to exclude.
        exclude_dim_red: Dimensionality reduction algorithms to exclude.
        merge_info0: Treat '_info0' variants as base model type.

    Returns:
        Dictionary containing:
        - 'df_summary': Polars DataFrame with formatted strings per cell
        - 'df_details': Polars DataFrame with granular statistics (delta, std,
          w_stat, p_raw, p_adj, sig)
        - 'models': List of model types
        - 'metrics': List of evaluated metrics
        - 'alpha': Alpha threshold used
        - 'alternative': Test direction used
        - 'correction': Correction method used
        - 'n_topic_blocks': Number of paired topic count blocks (N=5)
    """
    if df_default.is_empty() or df_alternative.is_empty():
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": [],
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_topic_blocks": 0,
        }

    target_metrics = [m for m in (metrics or METRICS)]

    # 1. Dataset normalization and filtering
    std_df = df_default.clone()
    alt_df = df_alternative.clone()

    for df_ref in [std_df, alt_df]:
        if "dataset_name" in df_ref.columns:
            df_ref = df_ref.with_columns(
                pl.col("dataset_name")
                .replace("anes_stemmed", "anes")
                .str.replace(r"_s\d+$", "")
            )

    if dataset:
        if "dataset_name" in std_df.columns:
            std_df = std_df.filter(pl.col("dataset_name") == dataset)
        if "dataset_name" in alt_df.columns:
            alt_df = alt_df.filter(pl.col("dataset_name") == dataset)

    if std_df.is_empty() or alt_df.is_empty():
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": target_metrics,
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_topic_blocks": 0,
        }

    # 2. Extract model_type and topic_idx
    def annotate_types_and_topics(df: pl.DataFrame) -> pl.DataFrame:
        m_types = []
        t_indices = []
        model_names = df["model_name"].to_list() if "model_name" in df.columns else []
        for name in model_names:
            mtype, tidx = parse_model_type_and_topic(str(name), merge_info0=merge_info0)
            m_types.append(mtype)
            t_indices.append(tidx)

        annotated = df.with_columns(
            [
                pl.Series("model_type", m_types),
                pl.Series("topic_idx", t_indices),
            ]
        )

        # Fallback for topic_idx if parsed as 0 but n_topics / nr_topics exists
        if "nr_topics" in annotated.columns:
            annotated = annotated.with_columns(
                pl.when(pl.col("topic_idx") == 0)
                .then(pl.col("nr_topics").fill_null(0))
                .otherwise(pl.col("topic_idx"))
                .alias("topic_idx")
            )
        elif "n_topics" in annotated.columns:
            annotated = annotated.with_columns(
                pl.when(pl.col("topic_idx") == 0)
                .then(pl.col("n_topics").fill_null(0))
                .otherwise(pl.col("topic_idx"))
                .alias("topic_idx")
            )

        return annotated

    std_df = annotate_types_and_topics(std_df)
    alt_df = annotate_types_and_topics(alt_df)

    # 3. Apply exclusion filters with normalized matching
    def filter_algos(
        df: pl.DataFrame,
        col_name: str,
        exclusions: Optional[List[str]],
        default_pattern: Optional[str],
    ) -> pl.DataFrame:
        if col_name not in df.columns:
            return df
        if exclusions is not None:
            norm_exclusions = [e.lower().replace("_", "") for e in exclusions]
            # Match if normalized value is in exclusions or contains any excluded token
            cond = (
                df[col_name]
                .fill_null("")
                .map_elements(
                    lambda x: any(
                        ex in x.lower().replace("_", "")
                        or x.lower().replace("_", "") == ex
                        for ex in norm_exclusions
                    ),
                    return_dtype=pl.Boolean,
                )
            )
            return df.filter(~cond)
        elif default_pattern:
            return df.filter(~pl.col(col_name).str.contains(default_pattern))
        return df

    std_df = filter_algos(std_df, "clustering_algo", exclude_clustering, "k_means")
    alt_df = filter_algos(alt_df, "clustering_algo", exclude_clustering, "k_means")

    std_df = filter_algos(std_df, "dim_red_algo", exclude_dim_red, "pca")
    alt_df = filter_algos(alt_df, "dim_red_algo", exclude_dim_red, "pca")

    if std_df.is_empty() or alt_df.is_empty():
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": target_metrics,
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_topic_blocks": 0,
        }

    available_metrics = [
        m for m in target_metrics if m in std_df.columns and m in alt_df.columns
    ]
    if not available_metrics:
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": [],
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_topic_blocks": 0,
        }

    # 4. Seed Aggregation: Average across seeds for each (model_type, topic_idx)
    agg_exprs = [
        pl.col(m).cast(pl.Float64, strict=False).mean().alias(m)
        for m in available_metrics
    ]

    std_agg = (
        std_df.group_by(["model_type", "topic_idx"])
        .agg(agg_exprs)
        .sort(["model_type", "topic_idx"])
    )
    alt_agg = (
        alt_df.group_by(["model_type", "topic_idx"])
        .agg(agg_exprs)
        .sort(["model_type", "topic_idx"])
    )

    # 5. Join default and alternative on paired (model_type, topic_idx)
    joined = std_agg.join(
        alt_agg, on=["model_type", "topic_idx"], suffix="_alternative"
    )
    if joined.is_empty():
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": available_metrics,
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_topic_blocks": 0,
        }

    # Desired canonical model order
    canonical_order = [
        "mv_co_reg_spectral",
        "mv_co_reg_spectral_info0",
        "mv_spectral",
        "mv_spectral_info0",
        "aligned_umap",
        "append_umap",
        "baseline",
        "umap_spectral",
        "stm",
    ]

    present_models = joined["model_type"].unique().to_list()
    ordered_models = [m for m in canonical_order if m in present_models]
    ordered_models += sorted([m for m in present_models if m not in canonical_order])

    # 6. Compute deltas and exact Wilcoxon test for each (model, metric)
    records = []
    for model in ordered_models:
        model_sub = joined.filter(pl.col("model_type") == model).sort("topic_idx")
        n_blocks = len(model_sub)

        for metric in available_metrics:
            alt_col = f"{metric}_alternative"
            d_series = (model_sub[alt_col] - model_sub[metric]).drop_nulls()
            d_vals = d_series.to_numpy()

            if len(d_vals) == 0:
                mean_d = 0.0
                std_d = 0.0
                w_stat = 0.0
                p_raw = 1.0
            else:
                mean_d = float(np.mean(d_vals))
                std_d = float(np.std(d_vals))
                w_stat, p_raw = wilcoxon_exact_test(
                    d_vals,
                    alternative=alternative,
                    zero_method=zero_method,
                )

            records.append(
                {
                    "model_type": model,
                    "metric": metric,
                    "mean_delta": mean_d,
                    "std_delta": std_d,
                    "n_blocks": n_blocks,
                    "w_stat": w_stat,
                    "p_raw": p_raw,
                }
            )

    # 7. Apply FWER Adjustment
    if correction == "per_metric":
        # Group by metric and apply Holm-Bonferroni per column
        for metric in available_metrics:
            metric_indices = [i for i, r in enumerate(records) if r["metric"] == metric]
            raw_ps = [records[i]["p_raw"] for i in metric_indices]
            adj_ps = holm_bonferroni(raw_ps)
            for i, adj_p in zip(metric_indices, adj_ps):
                records[i]["p_adj"] = adj_p
    elif correction == "table":
        # Global Holm-Bonferroni across all (model, metric) pairs
        raw_ps = [r["p_raw"] for r in records]
        adj_ps = holm_bonferroni(raw_ps)
        for r, adj_p in zip(records, adj_ps):
            r["p_adj"] = adj_p
    else:
        # No adjustment
        for r in records:
            r["p_adj"] = r["p_raw"]

    # 8. Add significance flags and cell formatting
    for r in records:
        is_sig = bool(r["p_adj"] < alpha)
        r["is_significant"] = is_sig
        flag = "*" if is_sig else ""
        mean_d = r["mean_delta"]
        # Format string e.g. +0.034* or -0.012
        r["formatted"] = f"{mean_d:+.3f}{flag}"

    df_details = pl.DataFrame(records)

    # 9. Build Summary Table (Rows: Models, Columns: Metrics)
    summary_rows = []
    for model in ordered_models:
        row_dict = {"Model": model}
        for metric in available_metrics:
            match = df_details.filter(
                (pl.col("model_type") == model) & (pl.col("metric") == metric)
            )
            if not match.is_empty():
                row_dict[metric] = match["formatted"][0]
            else:
                row_dict[metric] = "N/A"
        summary_rows.append(row_dict)

    df_summary = pl.DataFrame(summary_rows)

    return {
        "df_summary": df_summary,
        "df_details": df_details,
        "models": ordered_models,
        "metrics": available_metrics,
        "alpha": alpha,
        "alternative": alternative,
        "correction": correction,
        "n_topic_blocks": (
            df_details["n_blocks"].max() if not df_details.is_empty() else 0
        ),
    }
