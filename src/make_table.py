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
    highlight_colors: tuple[str, str, str] = ("FFD700", "C0C0C0", "CD7F32"),
    result_type: str | None = None,
) -> str:
    """
    Generates a consolidated LaTeX table from the best models analysis results.

    Args:
        results: Dictionary mapping metric names to Polars DataFrames of best models.
        dataset: Name of the dataset.
        dump: If True, uses model_name instead of model_type for rows.
        average: If True, indicates that values are averages of model runs.
        highlight_colors: Tuple of hex colors for 1st, 2nd, and 3rd best results.
        result_type: Optional result type identifier (e.g. 'standard',
            'stemmed', 'no_stopword_removal').

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
        "stm": "STM",
    }

    # 2. Build a matrix: rows are model types/names, columns are metrics
    # Explicitly order the IDs to match requirements
    desired_order = [
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
                mean_val = match["max_value"][0]
                std_val = match["std_value"][0] if "std_value" in match.columns else 0.0
                row[metric] = (mean_val, std_val)
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

    # Custom formatter for 3-tier coloring with mean +- std support
    def format_with_highlights(df):
        formatted_df = df.copy()
        for col in metric_cols:
            if col in df.columns:
                valid_entries = df[col].dropna()
                if not valid_entries.empty:
                    # Extract mean values for ranking
                    mean_vals = [
                        e[0] if isinstance(e, (tuple, list)) else e
                        for e in valid_entries
                    ]
                    top_vals = sorted(set(mean_vals), reverse=True)[:3]

                    def apply_color(entry):
                        if pd.isnull(entry):
                            return "-"
                        if isinstance(entry, (tuple, list)):
                            mean_val, std_val = entry
                        else:
                            mean_val, std_val = entry, 0.0

                        if pd.isnull(mean_val):
                            return "-"

                        if std_val is not None and std_val > 0.0:
                            val_str = f"${mean_val:.3f} \\pm {std_val:.3f}$"
                        else:
                            val_str = f"${mean_val:.3f}$"

                        if mean_val in top_vals:
                            rank = top_vals.index(mean_val)
                            if rank < len(highlight_colors):
                                color = highlight_colors[rank]
                                return f"\\cellcolor[HTML]{{{color}}}{{{val_str}}}"
                        return val_str

                    formatted_df[col] = df[col].apply(apply_color)
        return formatted_df

    display_df = format_with_highlights(final_df)
    display_df = display_df.rename(columns=actual_rename)

    # 6. Define caption based on result_type and dataset
    res_descr = ""
    if result_type:
        rt = result_type.lower()
        if rt in ("standard", "remove_rep_stopwords"):
            res_descr = " (unstemmed text with representation stopwords removed)"
        elif rt == "stemmed":
            res_descr = " (stemmed text with stopwords removed)"
        elif rt in (
            "no_stopword",
            "no_stopword_removal",
            "with_stopwords",
            "keep_rep_stopwords",
        ):
            res_descr = " (unstemmed text with representation stopwords kept)"

    if dump:
        caption = f"All model configurations for the {dataset} dataset{res_descr}."
    elif average:
        if result_type:
            rt = result_type.lower()
            if rt in ("standard", "remove_rep_stopwords"):
                avg_res_str = " with representation stopwords removed"
            elif rt == "stemmed":
                avg_res_str = " with stemmed text and stopwords removed"
            elif rt in (
                "no_stopword",
                "no_stopword_removal",
                "with_stopwords",
                "keep_rep_stopwords",
            ):
                avg_res_str = " with representation stopwords kept"
            else:
                avg_res_str = ""
        else:
            avg_res_str = ""
        caption = (
            f"Average performance by model type for the {dataset} dataset{avg_res_str} "
            "(results are reported as $\\text{mean} \\pm \\text{std}$ "
            "across random seeds)."
        )
    else:
        caption = (
            f"Best performing models by type for the {dataset} dataset{res_descr}."
        )

    caption += (
        f" \\textcolor[HTML]{{{highlight_colors[0]}}}{{1st}}, "
        f"\\textcolor[HTML]{{{highlight_colors[1]}}}{{2nd}}, and "
        f"\\textcolor[HTML]{{{highlight_colors[2]}}}{{3rd}} "
        "best results are highlighted."
    )

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


def generate_stopword_impact_latex_table(
    results: dict[str, pl.DataFrame],
    dataset: str = "fed",
    pos_color: str = "D4EDDA",
    neg_color: str = "F8D7DA",
) -> str:
    """Generates a LaTeX table showing the impact of stopword removal.

    Displays mean +- std of metric differences (standard - no_stopword).
    Cells are colored green for positive change (improvement) and red for negative.

    Args:
        results: Dictionary mapping metric names to Polars DataFrames containing
            columns ['model_type', 'mean_delta', 'std_delta', 'n_pairs'].
        dataset: Dataset identifier string.
        pos_color: Hex color string for cell background on positive change (green).
        neg_color: Hex color string for cell background on negative change (red).

    Returns:
        Formatted LaTeX table string.
    """
    import pandas as pd

    if not results:
        return ""

    MODEL_RENAME_MAP = {
        "append_umap": "Naive",
        "mv_co_reg_spectral": "$\\text{\\systemshort}_1$",
        "mv_co_reg_spectral_info0": "$\\text{\\systemshort}_1\\text{-info0}$",
        "baseline": "$\\text{BERTopic}_1$",
        "umap_spectral": "$\\text{BERTopic}_2$",
        "mv_spectral": "$\\text{\\systemshort}_2$",
        "mv_spectral_info0": "$\\text{\\systemshort}_2\\text{-info0}$",
        "aligned_umap": "$\\text{\\systemshort}_3$",
        "stm": "STM",
    }

    desired_order = [
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

    all_ids_present = set()
    for metric_df in results.values():
        if "model_type" in metric_df.columns:
            all_ids_present.update(metric_df["model_type"].to_list())

    all_ids = [i for i in desired_order if i in all_ids_present]
    all_ids += sorted(list(all_ids_present - set(desired_order)))

    rows = []
    for identifier in all_ids:
        display_name = MODEL_RENAME_MAP.get(identifier, identifier.replace("_", " "))
        row = {"Model Type": display_name}
        for metric, metric_df in results.items():
            match = metric_df.filter(pl.col("model_type") == identifier)
            if not match.is_empty():
                mean_val = match["mean_delta"][0]
                std_val = match["std_delta"][0] if "std_delta" in match.columns else 0.0
                row[metric] = (mean_val, std_val)
            else:
                row[metric] = None
        rows.append(row)

    final_df = pd.DataFrame(rows)
    metric_cols = [c for c in final_df.columns if c != "Model Type"]

    rename_map = {
        "u_mass": "$\\Delta C_{\\text{UMass}}$",
        "c_v": "$\\Delta C_v$",
        "c_npmi": "$\\Delta C_{npmi}$",
        "irbo": "$\\Delta \\text{IRBO}$",
        "topic_diversity": "$\\Delta \\text{Diversity}$",
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in final_df.columns}

    def format_impact_cells(df):
        formatted_df = df.copy()
        for col in metric_cols:
            if col in df.columns:

                def apply_color(entry):
                    if pd.isnull(entry):
                        return "-"
                    if isinstance(entry, (tuple, list)):
                        mean_val, std_val = entry
                    else:
                        mean_val, std_val = entry, 0.0

                    if pd.isnull(mean_val):
                        return "-"

                    sign = "+" if mean_val > 0 else ""
                    if std_val is not None and std_val > 0.0:
                        val_str = f"${sign}{mean_val:.3f} \\pm {std_val:.3f}$"
                    else:
                        val_str = f"${sign}{mean_val:.3f}$"

                    if mean_val > 0:
                        return f"\\cellcolor[HTML]{{{pos_color}}}{{{val_str}}}"
                    elif mean_val < 0:
                        return f"\\cellcolor[HTML]{{{neg_color}}}{{{val_str}}}"
                    return val_str

                formatted_df[col] = df[col].apply(apply_color)
        return formatted_df

    display_df = format_impact_cells(final_df)
    display_df = display_df.rename(columns=actual_rename)

    caption = (
        f"Average metric changes ($\\text{{mean}} \\pm \\text{{std}}$) resulting from "
        f"representation stopword removal for the {dataset} dataset "
        f"(comparing representation stopwords removed vs. kept). "
        f"\\cellcolor[HTML]{{{pos_color}}}{{Green}} indicates average "
        f"improvement ($\\Delta > 0$), and "
        f"\\cellcolor[HTML]{{{neg_color}}}{{red}} indicates average "
        f"decrease ($\\Delta < 0$)."
    )
    table_label = f"tab:stopword_impact_{dataset}"

    latex = display_df.to_latex(
        index=False,
        caption=caption,
        label=table_label,
        escape=False,
        column_format="l" + "r" * len(metric_cols),
        position="h!",
    )

    lines = latex.splitlines()
    processed_lines = []
    in_tabular = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("\\begin{table}"):
            processed_lines.append("\\begin{table}")
            processed_lines.append("\\centering")
            continue

        if (
            stripped.startswith("\\centering")
            or stripped.startswith("\\caption")
            or stripped.startswith("\\label")
        ):
            processed_lines.append(stripped)
            continue

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

        if in_tabular:
            processed_lines.append("            " + stripped)
        else:
            if stripped == "\\end{table}":
                processed_lines.append("\\end{table}")
            elif stripped:
                processed_lines.append(stripped)

    return "\n".join(processed_lines)


def generate_demsar_delta_markdown_table(
    delta_results: dict,
    dataset: str = "fed",
    condition_name: str = "Alternative",
) -> str:
    """Generates a Markdown table summarizing Demšar-compliant performance deltas.

    Args:
        delta_results: Output dictionary from compute_demsar_delta_table.
        dataset: Dataset identifier.
        condition_name: Name of the alternative condition (e.g., 'Stemmed').

    Returns:
        Formatted Markdown table string.
    """
    df_summary = delta_results.get("df_summary")
    if df_summary is None or df_summary.is_empty():
        return f"_No delta results available for dataset {dataset}_"

    metrics = delta_results.get("metrics", [])
    metric_labels = {
        "u_mass": "UMass",
        "c_v": "C_v",
        "c_npmi": "C_npmi",
        "irbo": "IRBO",
        "topic_diversity": "Diversity",
    }

    headers = ["Topic Model"] + [metric_labels.get(m, m) for m in metrics]
    col_align = [":---"] + [":---:"] * len(metrics)

    lines = []
    lines.append(
        f"### Performance Delta Table: {condition_name} vs. Default ({dataset.upper()})"
    )
    alpha = delta_results.get("alpha", 0.10)
    correction = delta_results.get("correction", "per_metric")
    lines.append(
        f"_Statistical significance tested via paired exact Wilcoxon "
        f"signed-rank test (N=5 topic counts) with Holm-Bonferroni "
        f"correction ({correction}, $\\alpha = {alpha}$). '*' denotes "
        f"adjusted $p < {alpha}$._\n"
    )

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(col_align) + " |")

    for row in df_summary.iter_rows(named=True):
        model_name = row["Model"]
        cells = [f"**{model_name}**"]
        for m in metrics:
            val = str(row.get(m, "N/A"))
            cells.append(val)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_demsar_delta_latex_table(
    delta_results: dict,
    dataset: str = "fed",
    condition_name: str = "Stemmed",
    pos_color: str = "D4EDDA",
    neg_color: str = "F8D7DA",
) -> str:
    """Generates a publication-ready LaTeX table for Demšar-compliant delta evaluations.

    Args:
        delta_results: Output dictionary from compute_demsar_delta_table.
        dataset: Dataset identifier string.
        condition_name: Description of the alternative condition.
        pos_color: Hex color for positive performance change.
        neg_color: Hex color for negative performance change.

    Returns:
        LaTeX table string with proper styling and sizing.
    """
    import pandas as pd

    df_details = delta_results.get("df_details")
    if df_details is None or df_details.is_empty():
        return ""

    MODEL_RENAME_MAP = {
        "append_umap": "Naive",
        "mv_co_reg_spectral": "$\\text{\\systemshort}_1$",
        "mv_co_reg_spectral_info0": "$\\text{\\systemshort}_1\\text{-info0}$",
        "baseline": "$\\text{BERTopic}_1$",
        "umap_spectral": "$\\text{BERTopic}_2$",
        "mv_spectral": "$\\text{\\systemshort}_2$",
        "mv_spectral_info0": "$\\text{\\systemshort}_2\\text{-info0}$",
        "aligned_umap": "$\\text{\\systemshort}_3$",
        "stm": "STM",
    }

    metrics = delta_results.get("metrics", [])
    models = delta_results.get("models", [])
    alpha = delta_results.get("alpha", 0.10)

    rows = []
    for model in models:
        display_name = MODEL_RENAME_MAP.get(model, model.replace("_", " "))
        row_dict = {"Model": display_name}
        for metric in metrics:
            match = df_details.filter(
                (pl.col("model_type") == model) & (pl.col("metric") == metric)
            )
            if not match.is_empty():
                mean_d = match["mean_delta"][0]
                std_d = match["std_delta"][0]
                is_sig = match["is_significant"][0]
                row_dict[metric] = (mean_d, std_d, is_sig)
            else:
                row_dict[metric] = None
        rows.append(row_dict)

    final_df = pd.DataFrame(rows)
    metric_cols = [c for c in final_df.columns if c != "Model"]

    rename_map = {
        "u_mass": "$\\Delta C_{\\text{UMass}}$",
        "c_v": "$\\Delta C_v$",
        "c_npmi": "$\\Delta C_{npmi}$",
        "irbo": "$\\Delta \\text{IRBO}$",
        "topic_diversity": "$\\Delta \\text{Diversity}$",
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in final_df.columns}

    def format_delta_cells(df):
        formatted_df = df.copy()
        for col in metric_cols:
            if col in df.columns:

                def apply_cell(entry):
                    if pd.isnull(entry) or entry is None:
                        return "-"
                    mean_val, std_val, is_sig = entry
                    if pd.isnull(mean_val):
                        return "-"

                    sign = "+" if mean_val > 0 else ""
                    star = "^{*}" if is_sig else ""

                    if std_val is not None and std_val > 0.0:
                        val_str = f"${sign}{mean_val:.3f} \\pm {std_val:.3f}{star}$"
                    else:
                        val_str = f"${sign}{mean_val:.3f}{star}$"

                    if mean_val > 0:
                        return f"\\cellcolor[HTML]{{{pos_color}}}{{{val_str}}}"
                    elif mean_val < 0:
                        return f"\\cellcolor[HTML]{{{neg_color}}}{{{val_str}}}"
                    return val_str

                formatted_df[col] = df[col].apply(apply_cell)
        return formatted_df

    display_df = format_delta_cells(final_df)
    display_df = display_df.rename(columns=actual_rename)

    caption = (
        f"Demšar-compliant Performance Delta Table for {condition_name} vs. "
        f"Default on the {dataset.upper()} dataset across $N=5$ topic counts. "
        f"Values indicate mean delta across topic counts "
        f"($\\text{{mean}} \\pm \\text{{std}}$). "
        f"Statistical significance tested via paired exact Wilcoxon signed-rank "
        f"tests with Holm-Bonferroni correction ($\\alpha = {alpha}$). "
        f"$^*$ denotes statistically significant difference "
        f"($p_{{\\text{{adj}}}} < {alpha}$)."
    )
    table_label = (
        f"tab:demsar_delta_{dataset}_{condition_name.lower().replace(' ', '_')}"
    )

    latex = display_df.to_latex(
        index=False,
        caption=caption,
        label=table_label,
        escape=False,
        column_format="l" + "r" * len(metric_cols),
        position="h!",
    )

    lines = latex.splitlines()
    processed_lines = []
    in_tabular = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("\\begin{table}"):
            processed_lines.append("\\begin{table}")
            processed_lines.append("\\centering")
            continue

        if (
            stripped.startswith("\\centering")
            or stripped.startswith("\\caption")
            or stripped.startswith("\\label")
        ):
            processed_lines.append(stripped)
            continue

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

        if in_tabular:
            processed_lines.append("            " + stripped)
        else:
            if stripped == "\\end{table}":
                processed_lines.append("\\end{table}")
            elif stripped:
                processed_lines.append(stripped)

    return "\n".join(processed_lines)


def generate_demsar_all_vs_all_markdown_table(
    all_vs_all_results: dict,
    metric: str,
    dataset_label: str = "",
) -> str:
    """Generates a Markdown Model Ranking Summary Table following Demšar (2006).

    Args:
        all_vs_all_results: Result dictionary from compute_demsar_all_vs_all.
        metric: Specific metric name to format.
        dataset_label: Optional label for the dataset/corpus.

    Returns:
        Formatted Markdown table string.
    """
    metrics_dict = all_vs_all_results.get("metrics", {})
    if metric not in metrics_dict:
        return f"Metric '{metric}' not found in results."

    m_data = metrics_dict[metric]
    df_summary = m_data.get("summary_table")
    if df_summary is None or df_summary.is_empty():
        return f"No summary data available for metric '{metric}'."

    omnibus = m_data.get("omnibus", {})
    f_stat = omnibus.get("f_f", 0.0)
    p_val = omnibus.get("p_f_f", 1.0)
    df1 = omnibus.get("df1", 0)
    df2 = omnibus.get("df2", 0)
    cd = m_data.get("critical_difference", 0.0)
    alpha = all_vs_all_results.get("metadata", {}).get("alpha", 0.05)
    n_blocks = m_data.get("n_blocks", 0)
    k_models = m_data.get("k_models", 0)

    metric_labels = {
        "u_mass": "Topic Coherence (U_Mass)",
        "c_v": "Topic Coherence (C_V)",
        "c_npmi": "Topic Coherence (NPMI)",
        "irbo": "Inverted RBO Diversity (IRBO)",
        "topic_diversity": "Topic Diversity",
    }
    m_title = metric_labels.get(metric, metric)
    d_title = f" [{dataset_label}]" if dataset_label else ""

    lines = []
    lines.append(f"### Demšar All-vs-All Ranking Summary: {m_title}{d_title}")
    sig_str = "Statistically Significant" if p_val < alpha else "Not Significant"
    lines.append(
        f"_Omnibus Iman-Davenport Test: $F_F({df1}, {df2}) = {f_stat:.3f}$, "
        f"$p = {p_val:.4f}$ ({sig_str} at $\\alpha = {alpha}$, "
        f"$N = {n_blocks}$ blocks, $k = {k_models}$ models). "
        f"Critical Difference (CD) = {cd:.3f}._\n"
    )

    headers = [
        "Model Name",
        "Mean Score (±SD)",
        "Mean Rank ($R_j$)",
        f"Significance Group ($\\alpha = {alpha}$)",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| :--- | :---: | :---: | :---: |")

    for row in df_summary.iter_rows(named=True):
        m_name = row["Model"]
        bold_prefix = "**" if row.get("Is Best", False) else ""
        bold_suffix = "**" if row.get("Is Best", False) else ""
        cells = [
            f"{bold_prefix}{m_name}{bold_suffix}",
            f"{row['Mean Score (±SD)']}",
            f"{row['Mean Rank']:.2f}",
            f"{row['Significance Group']}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_pairwise_delta_markdown_matrix(
    all_vs_all_results: dict,
    metric: str,
    dataset_label: str = "",
) -> str:
    """Generates a Markdown k x k Pairwise Delta Matrix following Demšar (2006).

    Args:
        all_vs_all_results: Result dictionary from compute_demsar_all_vs_all.
        metric: Specific metric name to format.
        dataset_label: Optional label for the dataset/corpus.

    Returns:
        Formatted Markdown table string.
    """
    metrics_dict = all_vs_all_results.get("metrics", {})
    if metric not in metrics_dict:
        return f"Metric '{metric}' not found in results."

    m_data = metrics_dict[metric]
    df_matrix = m_data.get("pairwise_delta_matrix")
    if df_matrix is None or df_matrix.is_empty():
        return f"No delta matrix available for metric '{metric}'."

    alpha = all_vs_all_results.get("metadata", {}).get("alpha", 0.05)
    metric_labels = {
        "u_mass": "U_Mass",
        "c_v": "C_V",
        "c_npmi": "NPMI",
        "irbo": "IRBO",
        "topic_diversity": "Diversity",
    }
    m_title = metric_labels.get(metric, metric)
    d_title = f" [{dataset_label}]" if dataset_label else ""

    lines = []
    lines.append(f"### Pairwise Delta Matrix: {m_title}{d_title}")
    lines.append(
        f"_Cell value: (Row Model Score - Column Model Score). "
        f"'*' indicates statistically significant difference after Holm-Bonferroni "
        f"post-hoc correction ($\\alpha = {alpha}$)._\n"
    )

    models = [c for c in df_matrix.columns if c != "Model"]
    headers = ["Model"] + models
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| :--- " + "| :---: " * len(models) + "|")

    for row in df_matrix.iter_rows(named=True):
        row_model = row["Model"]
        cells = [f"**{row_model}**"]
        for col_model in models:
            cells.append(str(row.get(col_model, "-")))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_demsar_all_vs_all_latex_table(
    all_vs_all_results: dict,
    metric: str,
    dataset_label: str = "FED",
) -> str:
    """Generates a publication-ready LaTeX table for Demšar All-vs-All Ranking Summary.

    Args:
        all_vs_all_results: Output dictionary from compute_demsar_all_vs_all.
        metric: Metric identifier string.
        dataset_label: Label for the dataset.

    Returns:
        LaTeX table string with proper styling.
    """
    import pandas as pd

    metrics_dict = all_vs_all_results.get("metrics", {})
    if metric not in metrics_dict:
        return ""

    m_data = metrics_dict[metric]
    df_summary = m_data.get("summary_table")
    if df_summary is None or df_summary.is_empty():
        return ""

    MODEL_RENAME_MAP = {
        "append_umap": "Naive",
        "mv_co_reg_spectral": "$\\text{\\systemshort}_1$",
        "mv_co_reg_spectral_info0": "$\\text{\\systemshort}_1\\text{-info0}$",
        "baseline": "$\\text{BERTopic}_1$",
        "umap_spectral": "$\\text{BERTopic}_2$",
        "mv_spectral": "$\\text{\\systemshort}_2$",
        "mv_spectral_info0": "$\\text{\\systemshort}_2\\text{-info0}$",
        "aligned_umap": "$\\text{\\systemshort}_3$",
        "stm": "STM",
    }

    omnibus = m_data.get("omnibus", {})
    f_stat = omnibus.get("f_f", 0.0)
    p_val = omnibus.get("p_f_f", 1.0)
    df1 = omnibus.get("df1", 0)
    df2 = omnibus.get("df2", 0)
    cd = m_data.get("critical_difference", 0.0)
    alpha = all_vs_all_results.get("metadata", {}).get("alpha", 0.05)

    rows = []
    for row in df_summary.iter_rows(named=True):
        m_name = row["Model"]
        display_name = MODEL_RENAME_MAP.get(m_name, m_name.replace("_", " "))
        if row.get("Is Best", False):
            display_name = f"\\textbf{{{display_name}}}"
        rows.append(
            {
                "Model": display_name,
                "Score": row["Mean Score (±SD)"],
                "Rank ($R_j$)": f"{row['Mean Rank']:.2f}",
                "Group": row["Significance Group"],
            }
        )

    pdf = pd.DataFrame(rows)

    metric_labels = {
        "u_mass": "$U_{Mass}$",
        "c_v": "$c_v$",
        "c_npmi": "$c_{npmi}$",
        "irbo": "IRBO",
        "topic_diversity": "Topic Diversity",
    }
    m_tex = metric_labels.get(metric, metric)

    caption = (
        f"Demšar (2006) All-vs-All Ranking Summary for {m_tex} ({dataset_label}). "
        f"Iman-Davenport omnibus test $F_F({df1}, {df2}) = {f_stat:.3f}$, "
        f"$p = {p_val:.4f}$. "
        f"Critical Difference $\\text{{CD}} = {cd:.3f}$ ($\\alpha = {alpha}$). "
        f"Models sharing a group letter are not significantly different."
    )
    label = f"tab:demsar_all_vs_all_{metric}_{dataset_label.lower().replace(' ', '_')}"

    latex = pdf.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=False,
        column_format="l c c c",
        position="h!",
    )

    lines = latex.splitlines()
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("\\begin{table}"):
            processed_lines.append("\\begin{table}")
            processed_lines.append("\\centering")
            continue
        if stripped.startswith("\\begin{tabular}"):
            processed_lines.append("\\resizebox{0.75\\columnwidth}{!}{%")
            processed_lines.append("    \\begin{tabular}" + stripped[15:])
            continue
        if stripped.startswith("\\end{tabular}"):
            processed_lines.append("    \\end{tabular}%")
            processed_lines.append("}")
            continue
        if stripped:
            processed_lines.append(stripped)

    return "\n".join(processed_lines)


def generate_pairwise_delta_latex_matrix(
    all_vs_all_results: dict,
    metric: str,
    dataset_label: str = "FED",
) -> str:
    """Generates a publication-ready LaTeX table for the k x k Pairwise Delta Matrix.

    Args:
        all_vs_all_results: Output dictionary from compute_demsar_all_vs_all.
        metric: Metric identifier string.
        dataset_label: Label for the dataset.

    Returns:
        LaTeX table string with proper styling.
    """
    import pandas as pd

    metrics_dict = all_vs_all_results.get("metrics", {})
    if metric not in metrics_dict:
        return ""

    m_data = metrics_dict[metric]
    df_matrix = m_data.get("pairwise_delta_matrix")
    if df_matrix is None or df_matrix.is_empty():
        return ""

    MODEL_RENAME_MAP = {
        "append_umap": "Naive",
        "mv_co_reg_spectral": "$\\text{\\systemshort}_1$",
        "mv_co_reg_spectral_info0": "$\\text{\\systemshort}_1\\text{-info0}$",
        "baseline": "$\\text{BERTopic}_1$",
        "umap_spectral": "$\\text{BERTopic}_2$",
        "mv_spectral": "$\\text{\\systemshort}_2$",
        "mv_spectral_info0": "$\\text{\\systemshort}_2\\text{-info0}$",
        "aligned_umap": "$\\text{\\systemshort}_3$",
        "stm": "STM",
    }

    models = [c for c in df_matrix.columns if c != "Model"]
    rows = []
    for row in df_matrix.iter_rows(named=True):
        m_name = row["Model"]
        display_name = MODEL_RENAME_MAP.get(m_name, m_name.replace("_", " "))
        row_dict = {"Model": display_name}
        for col_m in models:
            col_disp = MODEL_RENAME_MAP.get(col_m, col_m.replace("_", " "))
            val = str(row.get(col_m, "-"))
            if "*" in val:
                val = val.replace("*", "$^*$")
            row_dict[col_disp] = val
        rows.append(row_dict)

    pdf = pd.DataFrame(rows)

    metric_labels = {
        "u_mass": "$U_{Mass}$",
        "c_v": "$c_v$",
        "c_npmi": "$c_{npmi}$",
        "irbo": "IRBO",
        "topic_diversity": "Topic Diversity",
    }
    m_tex = metric_labels.get(metric, metric)
    alpha = all_vs_all_results.get("metadata", {}).get("alpha", 0.05)

    caption = (
        f"Demšar (2006) Pairwise Delta Matrix for {m_tex} ({dataset_label}). "
        f"Cells show $\\Delta\\text{{Score}} = \\text{{Row}} - \\text{{Column}}$. "
        f"$^*$ denotes statistically significant difference with "
        f"Holm-Bonferroni correction ($\\alpha = {alpha}$)."
    )
    label = (
        f"tab:demsar_pairwise_matrix_{metric}_{dataset_label.lower().replace(' ', '_')}"
    )

    col_format = "l " + "c " * len(models)
    latex = pdf.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=False,
        column_format=col_format,
        position="h!",
    )

    lines = latex.splitlines()
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("\\begin{table}"):
            processed_lines.append("\\begin{table}")
            processed_lines.append("\\centering")
            continue
        if stripped.startswith("\\begin{tabular}"):
            processed_lines.append("\\resizebox{\\columnwidth}{!}{%")
            processed_lines.append("    \\begin{tabular}" + stripped[15:])
            continue
        if stripped.startswith("\\end{tabular}"):
            processed_lines.append("    \\end{tabular}%")
            processed_lines.append("}")
            continue
        if stripped:
            processed_lines.append(stripped)

    return "\n".join(processed_lines)


def generate_demsar_all_vs_all_report(
    all_vs_all_results: dict,
    dataset_label: str = "",
    include_deltas: bool = False,
) -> str:
    """Generates a comprehensive Markdown report across all evaluated metrics.

    Args:
        all_vs_all_results: Result dictionary from compute_demsar_all_vs_all.
        dataset_label: Optional label for the dataset/corpus.
        include_deltas: Whether to include pairwise delta matrices in the report.

    Returns:
        Multi-section Markdown report string.
    """
    metrics_dict = all_vs_all_results.get("metrics", {})
    if not metrics_dict:
        return "No evaluation metrics found in results."

    metadata = all_vs_all_results.get("metadata", {})
    n_blocks = metadata.get("n_blocks", 0)
    k_models = metadata.get("k_models", 0)
    alpha = metadata.get("alpha", 0.05)
    datasets = metadata.get("datasets", [])

    ds_str = ", ".join(datasets) if datasets else (dataset_label or "Benchmark")

    report_lines = [
        "# Demšar (2006) All-vs-All Statistical Comparison Report",
        f"**Benchmark Dataset(s)**: {ds_str}  ",
        f"**Evaluation Blocks (N)**: {n_blocks} (Topic counts / configurations)  ",
        f"**Algorithms (k)**: {k_models}  ",
        f"**Significance Level (α)**: {alpha}  \n",
        "---",
    ]

    for metric_name in metrics_dict:
        summary_md = generate_demsar_all_vs_all_markdown_table(
            all_vs_all_results, metric=metric_name, dataset_label=dataset_label
        )
        report_lines.append(summary_md)
        if include_deltas:
            delta_md = generate_pairwise_delta_markdown_matrix(
                all_vs_all_results, metric=metric_name, dataset_label=dataset_label
            )
            report_lines.append("")
            report_lines.append(delta_md)
        report_lines.append("\n---\n")

    return "\n".join(report_lines)


# ==============================================================================
# Publication Table Helper Mappings & Dataframe/Styler Generators
# ==============================================================================

MODEL_LATEX_MAP = {
    "append_umap": "Naive",
    "mv_co_reg_spectral": r"$\text{\systemshort}_1$",
    "mv_co_reg_spectral_info0": r"$\text{\systemshort}_1\text{-info0}$",
    "baseline": r"$\text{BERTopic}_1$",
    "umap_spectral": r"$\text{BERTopic}_2$",
    "mv_spectral": r"$\text{\systemshort}_2$",
    "mv_spectral_info0": r"$\text{\systemshort}_2\text{-info0}$",
    "aligned_umap": r"$\text{\systemshort}_3$",
    "stm": "STM",
}

MODEL_DISPLAY_MAP = {
    "append_umap": "Naive",
    "mv_co_reg_spectral": "CAST₁",
    "mv_co_reg_spectral_info0": "CAST₁-info0",
    "baseline": "BERTopic₁",
    "umap_spectral": "BERTopic₂",
    "mv_spectral": "CAST₂",
    "mv_spectral_info0": "CAST₂-info0",
    "aligned_umap": "CAST₃",
    "stm": "STM",
}

MODEL_MARKDOWN_MAP = {
    "append_umap": "Naive",
    "mv_co_reg_spectral": r"$\text{CAST}_1$",
    "mv_co_reg_spectral_info0": r"$\text{CAST}_1\text{-info0}$",
    "baseline": r"$\text{BERTopic}_1$",
    "umap_spectral": r"$\text{BERTopic}_2$",
    "mv_spectral": r"$\text{CAST}_2$",
    "mv_spectral_info0": r"$\text{CAST}_2\text{-info0}$",
    "aligned_umap": r"$\text{CAST}_3$",
    "stm": "STM",
}

DESIRED_MODEL_ORDER = [
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

METRIC_LABEL_MAP = {
    "u_mass": "UMass",
    "c_v": "C_v",
    "c_npmi": "C_npmi",
    "irbo": "IRBO",
    "topic_diversity": "Diversity",
}


def generate_best_models_table_data(
    results: dict[str, pl.DataFrame],
    dump: bool = False,
    average: bool = False,
    highlight_colors: tuple[str, str, str] = ("#FFD700", "#C0C0C0", "#CD7F32"),
    model_name_map: dict[str, str] | None = None,
) -> dict:
    """Generates display DataFrame, numeric DataFrame, and Styler.

    Includes Olympic highlights for top 3 models per metric.

    Args:
        results: Dictionary mapping metric names to Polars DataFrames of best models.
        dump: If True, uses best_model_name instead of model_type for rows.
        average: If True, values represent averages with standard deviations.
        highlight_colors: Hex colors for 1st (Gold), 2nd (Silver), and 3rd (Bronze).
        model_name_map: Optional mapping for model names.

    Returns:
        Dict with keys:
            - 'display_df': Pandas DataFrame with formatted strings
            - 'numeric_df': Pandas DataFrame with numeric float mean values
            - 'styler': Pandas Styler with Olympic 3-tier cell backgrounds
            - 'metric_cols': List of metric column names present
    """
    import pandas as pd

    if not results:
        empty_df = pd.DataFrame()
        return {
            "display_df": empty_df,
            "numeric_df": empty_df,
            "styler": empty_df.style,
            "metric_cols": [],
        }

    id_col = "best_model_name" if dump else "model_type"
    all_ids_present = set()
    for metric_df in results.values():
        if id_col in metric_df.columns:
            all_ids_present.update(metric_df[id_col].to_list())

    all_ids = [i for i in DESIRED_MODEL_ORDER if i in all_ids_present]
    all_ids += sorted(list(all_ids_present - set(DESIRED_MODEL_ORDER)))

    mapping = model_name_map if model_name_map is not None else MODEL_DISPLAY_MAP

    display_rows = []
    numeric_rows = []
    metric_keys = list(results.keys())

    for identifier in all_ids:
        disp_name = mapping.get(identifier, identifier.replace("_", " "))
        disp_row = {"Model": disp_name}
        num_row = {"Model": disp_name}

        for metric in metric_keys:
            metric_df = results[metric]
            match = metric_df.filter(pl.col(id_col) == identifier)
            col_name = METRIC_LABEL_MAP.get(metric, metric)
            if not match.is_empty():
                mean_val = match["max_value"][0]
                std_val = match["std_value"][0] if "std_value" in match.columns else 0.0
                num_row[col_name] = mean_val
                if mean_val is None or (
                    isinstance(mean_val, float) and pd.isna(mean_val)
                ):
                    disp_row[col_name] = "-"
                elif std_val is not None and std_val > 0.0:
                    disp_row[col_name] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    disp_row[col_name] = f"{mean_val:.3f}"
            else:
                num_row[col_name] = None
                disp_row[col_name] = "-"

        display_rows.append(disp_row)
        numeric_rows.append(num_row)

    display_df = pd.DataFrame(display_rows)
    numeric_df = pd.DataFrame(numeric_rows)
    metric_cols = [c for c in display_df.columns if c != "Model"]

    def highlight_olympic(col_series):
        col_name = col_series.name
        if col_name not in numeric_df.columns or col_name == "Model":
            return [""] * len(col_series)

        valid_vals = numeric_df[col_name].dropna()
        if valid_vals.empty:
            return [""] * len(col_series)

        top_vals = sorted(valid_vals.unique(), reverse=True)[:3]
        styles = []
        for i in range(len(col_series)):
            num_val = numeric_df[col_name].iloc[i]
            if pd.isna(num_val):
                styles.append("")
            elif num_val in top_vals:
                rank = top_vals.index(num_val)
                if rank == 0 and len(highlight_colors) > 0:
                    styles.append(
                        f"background-color: {highlight_colors[0]}; "
                        "color: #000000; font-weight: bold;"
                    )
                elif rank == 1 and len(highlight_colors) > 1:
                    styles.append(
                        f"background-color: {highlight_colors[1]}; "
                        "color: #000000; font-weight: bold;"
                    )
                elif rank == 2 and len(highlight_colors) > 2:
                    styles.append(
                        f"background-color: {highlight_colors[2]}; "
                        "color: #FFFFFF; font-weight: bold;"
                    )
                else:
                    styles.append("")
            else:
                styles.append("")
        return styles

    styler = display_df.style.apply(highlight_olympic, axis=0)

    return {
        "display_df": display_df,
        "numeric_df": numeric_df,
        "styler": styler,
        "metric_cols": metric_cols,
    }


def generate_best_models_markdown_table(
    results: dict[str, pl.DataFrame],
    dataset: str = "fed",
    dump: bool = False,
    average: bool = False,
    result_type: str | None = None,
) -> str:
    """Generates a GitHub-Flavored Markdown table with bold styling for best models."""
    data = generate_best_models_table_data(
        results, dump=dump, average=average, model_name_map=MODEL_MARKDOWN_MAP
    )
    display_df = data["display_df"]
    numeric_df = data["numeric_df"]
    metric_cols = data["metric_cols"]

    if display_df.empty:
        return f"_No results available for dataset {dataset}_"

    lines = []
    title_suffix = f" ({result_type})" if result_type else ""
    if dump:
        lines.append(f"### All Model Configurations: {dataset.upper()}{title_suffix}")
    elif average:
        lines.append(f"### Average Model Performance: {dataset.upper()}{title_suffix}")
    else:
        lines.append(f"### Best Models by Type: {dataset.upper()}{title_suffix}")

    headers = ["Model"] + metric_cols
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| :--- " + "| :---: " * len(metric_cols) + "|")

    # Find top value per column for bolding
    top_per_col = {}
    for col in metric_cols:
        vals = numeric_df[col].dropna()
        if not vals.empty:
            top_per_col[col] = vals.max()

    for i in range(len(display_df)):
        row = display_df.iloc[i]
        m_name = row["Model"]
        cells = [f"**{m_name}**"]
        for col in metric_cols:
            val_str = str(row[col])
            num_val = numeric_df[col].iloc[i]
            if col in top_per_col and num_val == top_per_col[col]:
                cells.append(f"**{val_str}**")
            else:
                cells.append(val_str)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_stopword_impact_table_data(
    results: dict[str, pl.DataFrame],
    pos_color: str = "#D4EDDA",
    neg_color: str = "#F8D7DA",
    model_name_map: dict[str, str] | None = None,
) -> dict:
    """Generates display DataFrame, numeric DataFrame, and Styler for impact."""
    import pandas as pd

    if not results:
        empty_df = pd.DataFrame()
        return {
            "display_df": empty_df,
            "numeric_df": empty_df,
            "styler": empty_df.style,
            "metric_cols": [],
        }

    mapping = model_name_map if model_name_map is not None else MODEL_DISPLAY_MAP
    all_ids_present = set()
    for metric_df in results.values():
        if "model_type" in metric_df.columns:
            all_ids_present.update(metric_df["model_type"].to_list())

    all_ids = [i for i in DESIRED_MODEL_ORDER if i in all_ids_present]
    all_ids += sorted(list(all_ids_present - set(DESIRED_MODEL_ORDER)))

    display_rows = []
    numeric_rows = []
    metric_keys = list(results.keys())

    for identifier in all_ids:
        disp_name = mapping.get(identifier, identifier.replace("_", " "))
        disp_row = {"Model": disp_name}
        num_row = {"Model": disp_name}

        for metric in metric_keys:
            metric_df = results[metric]
            match = metric_df.filter(pl.col("model_type") == identifier)
            col_name = f"Δ {METRIC_LABEL_MAP.get(metric, metric)}"
            if not match.is_empty():
                mean_val = match["mean_delta"][0]
                std_val = match["std_delta"][0] if "std_delta" in match.columns else 0.0
                num_row[col_name] = mean_val
                sign = "+" if mean_val > 0 else ""
                if std_val is not None and std_val > 0.0:
                    disp_row[col_name] = f"{sign}{mean_val:.3f} ± {std_val:.3f}"
                else:
                    disp_row[col_name] = f"{sign}{mean_val:.3f}"
            else:
                num_row[col_name] = None
                disp_row[col_name] = "-"

        display_rows.append(disp_row)
        numeric_rows.append(num_row)

    display_df = pd.DataFrame(display_rows)
    numeric_df = pd.DataFrame(numeric_rows)
    metric_cols = [c for c in display_df.columns if c != "Model"]

    def style_deltas(col_series):
        col_name = col_series.name
        if col_name not in numeric_df.columns or col_name == "Model":
            return [""] * len(col_series)
        styles = []
        for i in range(len(col_series)):
            val = numeric_df[col_name].iloc[i]
            if pd.isna(val):
                styles.append("")
            elif val > 0:
                styles.append(
                    f"background-color: {pos_color}; color: #155724; font-weight: bold;"
                )
            elif val < 0:
                styles.append(
                    f"background-color: {neg_color}; color: #721C24; font-weight: bold;"
                )
            else:
                styles.append("")
        return styles

    styler = display_df.style.apply(style_deltas, axis=0)

    return {
        "display_df": display_df,
        "numeric_df": numeric_df,
        "styler": styler,
        "metric_cols": metric_cols,
    }


def generate_stopword_impact_markdown_table(
    results: dict[str, pl.DataFrame],
    dataset: str = "fed",
) -> str:
    """Generates a Markdown table for stopword removal impact."""
    data = generate_stopword_impact_table_data(
        results, model_name_map=MODEL_MARKDOWN_MAP
    )
    display_df = data["display_df"]
    metric_cols = data["metric_cols"]

    if display_df.empty:
        return f"_No stopword impact data available for {dataset}_"

    lines = [
        f"### Representation Stopword Impact (Δ Metric): {dataset.upper()}",
        (
            "_Values show difference: (Stopwords Removed - Stopwords Kept). "
            "Positive Δ indicates improvement._\n"
        ),
    ]
    headers = ["Model"] + metric_cols
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| :--- " + "| :---: " * len(metric_cols) + "|")

    for _, row in display_df.iterrows():
        cells = [f"**{row['Model']}**"] + [str(row[c]) for c in metric_cols]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_noise_coverage_latex_table(
    df: pl.DataFrame, result_type: str | None = None
) -> str:
    """Generates a publication-ready LaTeX table for HDBSCAN noise-cluster coverage."""
    res_descr = ""
    if result_type:
        rt = result_type.lower()
        if rt == "standard":
            res_descr = (
                " for Standard Unstemmed Text with Representation Stopwords Removed"
            )
        elif rt == "stemmed":
            res_descr = " for Stemmed Text with Stopwords Removed"
        elif rt in ("no_stopword", "no_stopword_removal", "with_stopwords"):
            res_descr = " for Unstemmed Text without Representation Stopword Removal"

    caption_text = (
        f"HDBSCAN Noise-Cluster Coverage across Random Seeds{res_descr} "
        r"(Mean $\pm$ Standard Deviation)"
    )
    col_header = (
        r"    Dataset & Model & Runs & Mean Noise Docs & "
        r"Mean Noise Coverage (\% $\pm$ SD) \\"
    )
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption_text}}}",
        r"\label{tab:noise_coverage}",
        r"\begin{tabular}{llrrr}",
        r"    \toprule",
        col_header,
        r"    \midrule",
    ]

    for row in df.iter_rows(named=True):
        dataset = row.get("dataset_name", "N/A")
        raw_m = str(row.get("model_type", row.get("model_name", "N/A")))
        model = MODEL_LATEX_MAP.get(raw_m, raw_m.replace("_", r"\_"))
        runs = row.get("n_runs", row.get("runs", 1))
        mean_outliers = row.get(
            "outliers_mean", row.get("mean_noise_docs", row.get("outliers", 0))
        )
        mean_pct = row.get(
            "noise_coverage_pct_mean",
            row.get("mean_noise_coverage_pct", row.get("noise_coverage_pct", 0.0)),
        )
        std_pct = row.get(
            "noise_coverage_pct_std", row.get("std_noise_coverage_pct", 0.0)
        )

        if std_pct is not None and std_pct > 0:
            pct_str = f"${mean_pct:.2f} \\pm {std_pct:.2f}$\\%"
        else:
            pct_str = f"{mean_pct:.2f}\\%"

        lines.append(
            f"    {dataset} & {model} & {runs} & {mean_outliers:.1f} & {pct_str} \\\\"
        )

    lines.extend([r"    \bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_noise_coverage_markdown_table(df: pl.DataFrame) -> str:
    """Generates a Markdown table for HDBSCAN noise-cluster coverage."""
    if df.is_empty():
        return "_No noise coverage data available._"

    lines = [
        "### HDBSCAN Noise-Cluster Coverage Across Random Seeds",
        "| Dataset | Model | Runs | Mean Noise Docs | Mean Noise Coverage (% ± SD) |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]

    for row in df.iter_rows(named=True):
        dataset = row.get("dataset_name", "N/A")
        raw_m = str(row.get("model_type", row.get("model_name", "N/A")))
        model = MODEL_DISPLAY_MAP.get(raw_m, raw_m.replace("_", " "))
        runs = row.get("n_runs", row.get("runs", 1))
        mean_outliers = row.get(
            "outliers_mean", row.get("mean_noise_docs", row.get("outliers", 0))
        )
        mean_pct = row.get(
            "noise_coverage_pct_mean",
            row.get("mean_noise_coverage_pct", row.get("noise_coverage_pct", 0.0)),
        )
        std_pct = row.get(
            "noise_coverage_pct_std", row.get("std_noise_coverage_pct", 0.0)
        )

        if std_pct is not None and std_pct > 0:
            pct_str = f"{mean_pct:.2f}% ± {std_pct:.2f}%"
        else:
            pct_str = f"{mean_pct:.2f}%"

        lines.append(
            f"| {dataset} | **{model}** | {runs} | {mean_outliers:.1f} | {pct_str} |"
        )

    return "\n".join(lines)


def style_demsar_delta_dataframe(
    df,
    pos_color: str = "#D4EDDA",
    neg_color: str = "#F8D7DA",
):
    """Applies green/red styling for Demšar delta table cells."""

    def style_cell(val):
        if not isinstance(val, str):
            return ""
        s = val.strip()
        if s.startswith("+"):
            return f"background-color: {pos_color}; color: #155724; font-weight: bold;"
        elif s.startswith("-"):
            return f"background-color: {neg_color}; color: #721C24; font-weight: bold;"
        return ""

    style_fn = getattr(df.style, "map", None) or getattr(df.style, "applymap")
    return style_fn(style_cell)


def style_demsar_pairwise_matrix(
    df,
    pos_color: str = "#D4EDDA",
    neg_color: str = "#F8D7DA",
):
    """Applies green/red styling for Demšar pairwise delta matrix."""

    def style_cell(val):
        if not isinstance(val, str):
            return ""
        s = val.strip()
        if s.startswith("+"):
            return f"background-color: {pos_color}; color: #155724; font-weight: bold;"
        elif s.startswith("-"):
            return f"background-color: {neg_color}; color: #721C24; font-weight: bold;"
        return ""

    style_fn = getattr(df.style, "map", None) or getattr(df.style, "applymap")
    return style_fn(style_cell)
