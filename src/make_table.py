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
) -> str:
    """Generates a comprehensive Markdown report across all evaluated metrics.

    Args:
        all_vs_all_results: Result dictionary from compute_demsar_all_vs_all.
        dataset_label: Optional label for the dataset/corpus.

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
        delta_md = generate_pairwise_delta_markdown_matrix(
            all_vs_all_results, metric=metric_name, dataset_label=dataset_label
        )
        report_lines.append(summary_md)
        report_lines.append("")
        report_lines.append(delta_md)
        report_lines.append("\n---\n")

    return "\n".join(report_lines)
