import polars as pl
from typing import Dict, List

METRICS = ["u_mass", "c_v", "c_npmi", "irbo", "topic_diversity"]

def extract_model_type(model_name: str) -> str:
    """Extracts the base model type from a model name (e.g., 'baseline_1' -> 'baseline')."""
    if not isinstance(model_name, str):
        return str(model_name)
    
    if "_" in model_name:
        parts = model_name.split("_")
        # If the last part is a number, remove it
        if parts[-1].isdigit():
            return "_".join(parts[:-1])
    return model_name

def find_best_models(
    df: pl.DataFrame, 
    dataset: str, 
    exclude_clustering: List[str] | None = None, 
    exclude_dim_red: List[str] | None = None,
    dump: bool = False
) -> Dict[str, pl.DataFrame]:
    """
    Finds the best performing model of each model type for each metric in the given dataset.
    
    Args:
        df: Polars DataFrame containing experiment results.
        dataset: The name of the dataset to filter by.
        exclude_clustering: Optional list of clustering algorithms to exclude.
        exclude_dim_red: Optional list of dimensionality reduction algorithms to exclude.
        dump: If True, returns all model configurations instead of just the best per type.
        
    Returns:
        A dictionary where keys are metric names and values are DataFrames with models and their scores.
    """
    # Filter by dataset
    if "dataset_name" in df.columns:
        df = df.filter(pl.col("dataset_name") == dataset)
    
    if df.is_empty():
        return {}

    # Apply exclusion filters
    if exclude_clustering:
        df = df.filter(~pl.col("clustering_algo").is_in(exclude_clustering))
    
    if exclude_dim_red:
        df = df.filter(~pl.col("dim_red_algo").is_in(exclude_dim_red))

    if df.is_empty():
        return {}

    # Add model_type column if it doesn't exist
    if "model_type" not in df.columns:
        df = df.with_columns(
            pl.col("model_name").map_elements(extract_model_type, return_dtype=pl.String).alias("model_type")
        )
    
    available_metrics = [m for m in METRICS if m in df.columns]
    
    results = {}
    for metric in available_metrics:
        # Ensure metric is numeric and filter out nulls
        if df[metric].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            continue
            
        metric_df = df.filter(pl.col(metric).is_not_null())
        if metric_df.is_empty():
            continue

        if dump:
            # For dump mode, just sort by the metric and keep everything
            best_per_type = (
                metric_df.sort(metric, descending=True)
                .select([
                    pl.col("model_name").alias("best_model_name"),
                    pl.col(metric).alias("max_value"),
                    pl.col("model_type")
                ])
            )
        else:
            # For each metric, group by model_type and find the max
            best_per_type = (
                metric_df.sort(metric, descending=True)
                .group_by("model_type")
                .agg(
                    pl.col(metric).first().alias("max_value"),
                    pl.col("model_name").first().alias("best_model_name")
                )
                .sort("max_value", descending=True)
            )
        results[metric] = best_per_type
        
    return results
