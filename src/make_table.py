import polars as pl
from great_tables import GT, style, loc


def generate_gt_table(df: pl.DataFrame) -> GT:
    """
    Generates a Great Tables object from an experiment results DataFrame.
    """
    # 1. Preprocessing with Polars
    processed_df = df.clone()

    # Cast n_clusters to Int64 if it exists
    if "n_clusters" in processed_df.columns:
        processed_df = processed_df.with_columns(
            pl.col("n_clusters").cast(pl.Int64, strict=False)
        )

    # Prepare for display
    display_df = processed_df.with_columns(
        pl.col("model_name").str.replace_all("_", " ")
    )

    # Core columns to show
    core_cols = [
        "model_name", "dataset_name", "timestamp", "n_observations",
        "clustering_algo", "dim_red_algo", "n_topics"
    ]
    
    # Identify metric columns (everything else that is numeric)
    exclude_from_metrics = core_cols + ["duration_seconds", "outliers"]
    metric_cols = [
        col for col in display_df.columns 
        if col not in exclude_from_metrics and display_df[col].dtype in [pl.Float64, pl.Float32]
    ]

    # Final selection and ordering
    final_cols = core_cols + metric_cols
    display_df = display_df.select([c for c in final_cols if c in display_df.columns])

    # 2. Create Great Table
    gt_table = (
        GT(display_df.to_pandas())
        .tab_header(
            title="BERTopic Experiment Results",
            subtitle="Comparison of topic modeling configurations and metrics"
        )
        .fmt_number(
            columns=metric_cols,
            decimals=3
        )
        .cols_label(
            model_name="Model",
            dataset_name="Dataset",
            timestamp="Executed At",
            n_observations="Obs",
            clustering_algo="Clustering",
            dim_red_algo="Dim Red",
            n_topics="Topics"
        )
        .tab_options(
            table_font_size="smaller",
            column_labels_font_weight="bold"
        )
    )

    return gt_table


def generate_latex_table(df: pl.DataFrame) -> str:
    """
    Generates a LaTeX table from an experiment results DataFrame.
    """
    # Ensure n_clusters is Int64 for consistency
    if "n_clusters" in df.columns:
        df = df.with_columns(pl.col("n_clusters").cast(pl.Int64, strict=False))

    renamed_df = df.with_columns(
        pl.col("model_name").str.replace_all("_", " ")
    ).drop([
        "outliers",
        "duration_seconds"
    ]).rename({
        "model_name": "Model",
        "n_topics": "Topics",
        "u_mass": "$U_{Mass}$",
        "c_v": "$c_v$",
        "c_npmi": "$c_{npmi}$",
        "irbo": "IRBO",
        "topic_diversity": "Diversity"
    })

    # Filter to only existing columns in the rename map + core ones
    cols_to_keep = ["Model", "Topics", "$U_{Mass}$", "$c_v$", "$c_{npmi}$", "IRBO", "Diversity"]
    final_df = renamed_df.select([c for c in cols_to_keep if c in renamed_df.columns])

    return final_df.to_pandas().to_latex(index=False, float_format="%.3f")


def generate_best_models_latex_table(results: dict[str, pl.DataFrame], dataset: str) -> str:
    """
    Generates a consolidated LaTeX table from the best models analysis results.

    Args:
        results: Dictionary mapping metric names to Polars DataFrames of best models.
        dataset: Name of the dataset.

    Returns:
        A LaTeX table string.
    """
    import pandas as pd

    if not results:
        return ""

    # 1. Gather all unique model types present in any of the metric results
    all_model_types = set()
    for metric_df in results.values():
        all_model_types.update(metric_df["model_type"].to_list())
    
    all_model_types = sorted(list(all_model_types))

    # 2. Build a matrix: rows are model types, columns are metrics
    rows = []
    for mt in all_model_types:
        row = {"Model Type": mt.replace("_", " ")}
        for metric, metric_df in results.items():
            # Find the max_value for this specific model_type
            match = metric_df.filter(pl.col("model_type") == mt)
            if not match.is_empty():
                row[metric] = match["max_value"][0]
            else:
                row[metric] = None
        rows.append(row)

    # 3. Create Pandas DataFrame for easy LaTeX export
    final_df = pd.DataFrame(rows)

    # 4. Rename columns for a professional LaTeX look
    rename_map = {
        "u_mass": "$U_{Mass}$",
        "c_v": "$c_v$",
        "c_npmi": "$c_{npmi}$",
        "irbo": "IRBO",
        "topic_diversity": "Diversity"
    }
    # Only rename if the column exists
    actual_rename = {k: v for k, v in rename_map.items() if k in final_df.columns}
    final_df = final_df.rename(columns=actual_rename)

    # 5. Export to LaTeX
    latex = final_df.to_latex(
        index=False,
        float_format="%.3f",
        caption=f"Best performing models by type for the {dataset} dataset.",
        label=f"tab:best_models_{dataset}",
        na_rep="-",
        escape=False # Allow LaTeX math in headers
    )

    return latex
