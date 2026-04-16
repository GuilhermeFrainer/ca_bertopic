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


def generate_best_models_latex_table(results: dict[str, pl.DataFrame], dataset: str, dump: bool = False) -> str:
    """
    Generates a consolidated LaTeX table from the best models analysis results.

    Args:
        results: Dictionary mapping metric names to Polars DataFrames of best models.
        dataset: Name of the dataset.
        dump: If True, uses model_name instead of model_type for rows.

    Returns:
        A LaTeX table string.
    """
    import pandas as pd
    import numpy as np

    if not results:
        return ""

    # 1. Gather all unique identifiers (model_type or model_name)
    id_col = "best_model_name" if dump else "model_type"
    all_ids = set()
    for metric_df in results.values():
        all_ids.update(metric_df[id_col].to_list())
    
    all_ids = sorted(list(all_ids))

    # 2. Build a matrix: rows are model types/names, columns are metrics
    rows = []
    for identifier in all_ids:
        display_name = identifier.replace("_", " ")
        row = {"Model Type" if not dump else "Model": display_name}
        for metric, metric_df in results.items():
            # Find the value for this specific identifier
            match = metric_df.filter(pl.col(id_col) == identifier)
            if not match.is_empty():
                row[metric] = match["max_value"][0]
            else:
                row[metric] = None
        rows.append(row)

    # 3. Create Pandas DataFrame
    final_df = pd.DataFrame(rows)

    # 4. Identify best values for bolding
    metric_cols = [c for c in final_df.columns if c not in ["Model Type", "Model"]]
    
    # 5. Export to LaTeX with specific formatting
    rename_map = {
        "u_mass": "$C_{\\text{UMass}}$",
        "c_v": "$C_v$",
        "c_npmi": "$C_{npmi}$",
        "irbo": "IRBO",
        "topic_diversity": "Diversity"
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in final_df.columns}
    
    # We'll use a custom formatter for bolding
    def format_with_bold(df):
        formatted_df = df.copy()
        for col in metric_cols:
            if col in df.columns:
                valid_vals = df[col].dropna()
                if not valid_vals.empty:
                    max_val = valid_vals.max()
                    formatted_df[col] = df[col].apply(
                        lambda x: f"\\textbf{{{x:.3f}}}" if pd.notnull(x) and x == max_val else (f"{x:.3f}" if pd.notnull(x) else "-")
                    )
        return formatted_df

    display_df = format_with_bold(final_df)
    display_df = display_df.rename(columns=actual_rename)

    latex = display_df.to_latex(
        index=False,
        caption=f"Best performing models by type for the {dataset} dataset." if not dump else f"All model configurations for the {dataset} dataset.",
        label=f"tab:best_models_{dataset}" if not dump else f"tab:all_models_{dataset}",
        escape=False,
        column_format="l" + "r" * len(metric_cols),
        position="h!"
    )

    # Wrap in centering and resizebox
    wrapped_latex = "\\begin{table}\n    \\centering\n"
    wrapped_latex += f"    \\caption{{{display_df.attrs.get('caption', '')}}}\n" # to_latex already adds caption if provided, but we want custom wrapping
    
    # Need to clean up the to_latex output to fit inside our manual table environment
    # or just use the generated string and wrap it.
    
    # simpler approach: just use replace to add the wrappers
    latex = latex.replace("\\begin{table}", "\\begin{table}\n    \\centering")
    latex = latex.replace("\\begin{tabular}", "    \\resizebox{\\columnwidth}{!}{%\n        \\begin{tabular}")
    latex = latex.replace("\\end{tabular}", "        \\end{tabular}%\n    }")
    
    return latex
