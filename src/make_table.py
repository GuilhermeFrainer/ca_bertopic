import polars as pl
from great_tables import GT


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
        "model_name",
        "dataset_name",
        "timestamp",
        "n_observations",
        "clustering_algo",
        "dim_red_algo",
        "n_topics",
    ]

    # Identify metric columns (everything else that is numeric)
    exclude_from_metrics = core_cols + ["duration_seconds", "outliers"]
    metric_cols = [
        col
        for col in display_df.columns
        if col not in exclude_from_metrics
        and display_df[col].dtype in [pl.Float64, pl.Float32]
    ]

    # Final selection and ordering
    final_cols = core_cols + metric_cols
    display_df = display_df.select([c for c in final_cols if c in display_df.columns])

    # 2. Create Great Table
    gt_table = (
        GT(display_df.to_pandas())
        .tab_header(
            title="BERTopic Experiment Results",
            subtitle="Comparison of topic modeling configurations and metrics",
        )
        .fmt_number(columns=metric_cols, decimals=3)
        .cols_label(
            model_name="Model",
            dataset_name="Dataset",
            timestamp="Executed At",
            n_observations="Obs",
            clustering_algo="Clustering",
            dim_red_algo="Dim Red",
            n_topics="Topics",
        )
        .tab_options(table_font_size="smaller", column_labels_font_weight="bold")
    )

    return gt_table


def generate_latex_table(df: pl.DataFrame) -> str:
    """
    Generates a LaTeX table from an experiment results DataFrame.
    """
    # Ensure n_clusters is Int64 for consistency
    if "n_clusters" in df.columns:
        df = df.with_columns(pl.col("n_clusters").cast(pl.Int64, strict=False))

    renamed_df = (
        df.with_columns(pl.col("model_name").str.replace_all("_", " "))
        .drop(["outliers", "duration_seconds"])
        .rename(
            {
                "model_name": "Model",
                "n_topics": "Topics",
                "u_mass": "$U_{Mass}$",
                "c_v": "$c_v$",
                "c_npmi": "$c_{npmi}$",
                "irbo": "IRBO",
                "topic_diversity": "Diversity",
            }
        )
    )

    # Filter to only existing columns in the rename map + core ones
    cols_to_keep = [
        "Model",
        "Topics",
        "$U_{Mass}$",
        "$c_v$",
        "$c_{npmi}$",
        "IRBO",
        "Diversity",
    ]
    final_df = renamed_df.select([c for c in cols_to_keep if c in renamed_df.columns])

    return final_df.to_pandas().to_latex(index=False, float_format="%.3f")


def generate_best_models_latex_table(
    results: dict[str, pl.DataFrame],
    dataset: str,
    dump: bool = False,
    average: bool = False,
) -> str:
    """
    Generates a consolidated LaTeX table from the best models analysis results.

    Args:
        results: Dictionary mapping metric names to Polars DataFrames of best models.
        dataset: Name of the dataset.
        dump: If True, uses model_name instead of model_type for rows.
        average: If True, indicates that values are averages of model runs.

    Returns:
        A LaTeX table string.
    """
    import pandas as pd

    if not results:
        return ""

    # 1. Gather all unique identifiers (model_type or model_name)
    id_col = "best_model_name" if dump else "model_type"
    all_ids = set()
    for metric_df in results.values():
        all_ids.update(metric_df[id_col].to_list())

    all_ids = sorted(list(all_ids))

    # Model renaming for LaTeX
    MODEL_RENAME_MAP = {
        "append_umap": "Naive",
        "mv_co_reg_spectral": "$\\text{\\systemshort}_1$",
        "mv_co_reg_spectral_info0": "$\\text{\\systemshort}_1\\text{-info0}$",
        "baseline": "$\\text{BERTopic}_1$",
        "umap_spectral": "$\\text{BERTopic}_2$",
        "mv_spectral": "$\\text{\\systemshort}_2$",
        "mv_spectral_info0": "$\\text{\\systemshort}_2\\text{-info0}$",
        "aligned_umap": "$\\text{\\systemshort}_3$",
    }

    # 2. Build a matrix: rows are model types/names, columns are metrics
    # Explicitly order the IDs to match requirements
    desired_order = [
        "mv_co_reg_spectral",
        "mv_co_reg_spectral_info0",
        "mv_spectral",
        "mv_spectral_info0",
        "aligned_umap",
        "baseline",
        "umap_spectral",
        "append_umap",
    ]
    # Filter to only those present in the results
    all_ids_present = set()
    for metric_df in results.values():
        all_ids_present.update(metric_df[id_col].to_list())

    all_ids = [i for i in desired_order if i in all_ids_present]
    # Add any others that might be missing from desired_order but are in results
    all_ids += sorted(list(all_ids_present - set(desired_order)))

    rows = []
    for identifier in all_ids:
        display_name = MODEL_RENAME_MAP.get(identifier, identifier.replace("_", " "))
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

    # 4. Identify metric columns
    metric_cols = [c for c in final_df.columns if c not in ["Model Type", "Model"]]

    # 5. Export to LaTeX with specific formatting
    rename_map = {
        "u_mass": "$C_{\\text{UMass}}$",
        "c_v": "$C_v$",
        "c_npmi": "$C_{npmi}$",
        "irbo": "IRBO",
        "topic_diversity": "Diversity",
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in final_df.columns}

    # Custom formatter for 3-tier coloring
    def format_with_highlights(df):
        formatted_df = df.copy()
        for col in metric_cols:
            if col in df.columns:
                valid_vals = df[col].dropna()
                if not valid_vals.empty:
                    # Get top 3 unique values
                    top_vals = sorted(valid_vals.unique(), reverse=True)[:3]

                    def apply_color(x):
                        if pd.isnull(x):
                            return "-"
                        if x in top_vals:
                            rank = top_vals.index(x)
                            if rank == 0:
                                color = "B2E0B2"  # Strongest
                            elif rank == 1:
                                color = "CCE8CC"  # Medium
                            else:
                                color = "E5F5E5"  # Pale
                            return f"\\cellcolor[HTML]{{{color}}}{x:.3f}"
                        return f"{x:.3f}"

                    formatted_df[col] = df[col].apply(apply_color)
        return formatted_df

    display_df = format_with_highlights(final_df)
    display_df = display_df.rename(columns=actual_rename)

    # 6. Define caption
    if dump:
        caption = f"All model configurations for the {dataset} dataset."
    elif average:
        caption = (
            f"Average performance by model type for the {dataset} dataset "
            "(results are averaged across different model runs)."
        )
    else:
        caption = f"Best performing models by type for the {dataset} dataset."

    # Export to LaTeX
    if dump:
        table_label = f"tab:all_models_{dataset}"
    elif average:
        table_label = f"tab:avg_models_{dataset}"
    else:
        table_label = f"tab:best_models_{dataset}"

    latex = display_df.to_latex(
        index=False,
        caption=caption,
        label=table_label,
        escape=False,
        column_format="l" + "r" * len(metric_cols),
        position="h!",
    )

    # Custom post-processing for indentation and wrapping
    lines = latex.splitlines()
    processed_lines = []
    in_tabular = False

    for line in lines:
        stripped = line.strip()

        # 1. Handle table environment wrapping and centering
        if stripped.startswith("\\begin{table}"):
            processed_lines.append("\\begin{table}")
            processed_lines.append("\\centering")
            continue

        if (
            stripped.startswith("\\centering")
            or stripped.startswith("\\caption")
            or stripped.startswith("\\label")
        ):
            # Re-add these without indentation at the root level of the
            # table environment
            processed_lines.append(stripped)
            continue

        # 2. Handle resizebox and tabular indentation
        if stripped.startswith("\\begin{tabular}"):
            processed_lines.append("\\resizebox{\\columnwidth}{!}{%")
            processed_lines.append("    \\begin{tabular}" + stripped[15:])
            in_tabular = True
            continue

        if stripped.startswith("\\end{tabular}"):
            processed_lines.append("    \\end{tabular}%")
            processed_lines.append("}")
            in_tabular = False
            continue

        # 3. Indent content within tabular
        if in_tabular:
            # Three levels deep (12 spaces)
            processed_lines.append("            " + stripped)
        else:
            # Other lines (like \toprule outside, or \end{table})
            if stripped == "\\end{table}":
                processed_lines.append("\\end{table}")
            elif stripped:
                processed_lines.append(stripped)

    return "\n".join(processed_lines)
