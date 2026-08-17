"""Interactive dashboard for BERTopic experiment results analysis.

This module provides a Streamlit-based dashboard to load, filter, and visualize
experimental results from CSV files in the results directory.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import datetime
import glob
import json
import os
import re
from typing import Optional

import altair as alt
import polars as pl
import streamlit as st

from src.experiment_tracker import build_coverage_matrix, scan_experiment_configs

# Default metrics for the visualization
DEFAULT_X_AXIS = "u_mass"
DEFAULT_Y_AXIS = "irbo"

# Configuration for metrics: which direction is "better"
# This can be easily extended in the future.
METRIC_CONFIG = {
    "duration_seconds": "min",
    "c_v": "max",
    "c_npmi": "max",
    "u_mass": "max",  # Higher (closer to 0) is better
    "irbo": "max",
    "topic_diversity": "max",
    "n_topics": "max",
    "n_observations": "max",
}


def extract_model_type(name: str) -> str:
    """Extracts the base model type from a model name (e.g., baseline_1 -> baseline)."""
    if not name or not isinstance(name, str):
        return "unknown"

    res = name
    # Strip common prefixes
    if res.startswith("stemmed_"):
        res = res[len("stemmed_") :]

    # Handles common patterns like 'baseline_1' or 'mv_spectral_2'
    # We take the part before the last underscore if it's followed by a digit
    parts = res.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return res


@st.cache_data
def load_all_results(results_dir: str = "results") -> pl.DataFrame:
    """Loads and concatenates result files. Extracts dataset and date objects."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    all_files = csv_files + json_files

    if not all_files:
        return pl.DataFrame()

    dfs = []
    for file in all_files:
        try:
            if file.endswith(".csv"):
                df = pl.read_csv(file, infer_schema_length=None)
            elif file.endswith(".json"):
                df = pl.read_json(file, infer_schema_length=None)
            else:
                continue

            # Normalize dataset and model names early
            if "dataset_name" in df.columns:
                df = df.with_columns(
                    pl.col("dataset_name").replace("anes_stemmed", "anes")
                )

            if "model_name" in df.columns:
                df = df.with_columns(pl.col("model_name").str.replace("^stemmed_", ""))
            elif "model_id" in df.columns:
                df = df.with_columns(pl.col("model_id").str.replace("^stemmed_", ""))

            file_basename = os.path.basename(file)

            # Extract Date as actual date object
            date_match = re.search(r"-(\d{8})-", file_basename)
            exp_date = None
            if date_match:
                d_str = date_match.group(1)
                try:
                    exp_date = datetime.date(
                        int(d_str[:4]), int(d_str[4:6]), int(d_str[6:])
                    )
                except ValueError:
                    exp_date = None

            # Fallback to timestamp column if exp_date is still None
            if exp_date is None and "timestamp" in df.columns and len(df) > 0:
                ts_val = df["timestamp"][0]
                if ts_val and isinstance(ts_val, str):
                    try:
                        # Handle ISO formats like 2026-04-02T13:05:32.462016
                        exp_date = datetime.datetime.fromisoformat(ts_val).date()
                    except (ValueError, TypeError):
                        pass

            # Dataset extraction (New logic: use dataset_name column if it exists)
            dataset = "unknown"
            if "dataset_name" in df.columns and len(df) > 0:
                dataset = df["dataset_name"][0]
            elif "trump" in file_basename.lower():
                dataset = "trump"
            elif "yelp" in file_basename.lower():
                dataset = "yelp"
            else:
                dataset = file_basename.split("-")[0].split("_")[0]

            # Legacy fallback: ensure _embeddings suffix is stripped
            dataset = dataset.replace("_embeddings", "")

            df = df.with_columns(
                pl.lit(os.path.splitext(file_basename)[0]).alias("source_file"),
                pl.lit(dataset).alias("dataset_label"),
                pl.lit(exp_date).cast(pl.Date).alias("experiment_date"),
                pl.lit(
                    "optimizer" if "opt" in file_basename.lower() else "non-optimizer"
                ).alias("experiment_type"),
            )

            if "model_name" in df.columns:
                df = df.with_columns(
                    pl.col("model_name")
                    .map_elements(extract_model_type, return_dtype=pl.String)
                    .alias("model_type")
                )
            elif "model_id" in df.columns:
                df = df.with_columns(
                    pl.col("model_id")
                    .map_elements(extract_model_type, return_dtype=pl.String)
                    .alias("model_type")
                )
            else:
                df = df.with_columns(pl.lit("unknown").alias("model_type"))

            # Standardize complex columns across files to prevent schema mismatches
            if "representation" in df.columns:
                dtype = df["representation"].dtype
                if dtype == pl.String or dtype == pl.Utf8:

                    def parse_repr(x):
                        if not isinstance(x, str) or not x.strip():
                            return []
                        if x.startswith("["):
                            try:
                                return [str(w) for w in json.loads(x)]
                            except Exception:
                                pass
                        return [w.strip() for w in x.split(",")]

                    df = df.with_columns(
                        pl.col("representation").map_elements(
                            parse_repr, return_dtype=pl.List(pl.String)
                        )
                    )

            if "representative_docs" in df.columns:
                dtype = df["representative_docs"].dtype
                if dtype != pl.List(pl.String):
                    if isinstance(dtype, pl.List):
                        df = df.with_columns(
                            pl.col("representative_docs").cast(pl.List(pl.String))
                        )
                    else:
                        df = df.with_columns(
                            pl.col("representative_docs").map_elements(
                                lambda x: [str(x)] if x is not None else [],
                                return_dtype=pl.List(pl.String),
                            )
                        )

            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="diagonal_relaxed")


def main():
    """Main function to run the Streamlit dashboard."""
    st.set_page_config(
        page_title="CA-BERTopic Experiment Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 CA-BERTopic Experiment Dashboard")

    # 1. Load Data
    results_dir = "results"
    output_dir = "output"
    df = load_all_results(results_dir)
    qual_df = load_all_results(output_dir)

    if df.is_empty():
        st.warning(f"No result files found in `{results_dir}/`.")
        return

    # 2. Sidebar Filters
    st.sidebar.header("Data Filters")

    # --- Cascading Filter Logic Setup ---
    # We use session state to track selections and allow bidirectional filtering
    filter_config = {
        "dataset_label": {"label": "Datasets:", "is_sidebar": True},
        "model_type": {"label": "Model Types:", "is_sidebar": True},
        "experiment_type": {"label": "Experiment Types:", "is_sidebar": True},
        "clustering_algo": {"label": "Clustering Algo:", "is_sidebar": False},
        "dim_red_algo": {"label": "Dim Red Algo:", "is_sidebar": False},
        "n_observations": {"label": "N Observations:", "is_sidebar": False},
    }

    for key in filter_config:
        if key not in st.session_state:
            st.session_state[key] = []

    def get_filtered_df(exclude_key: Optional[str] = None) -> pl.DataFrame:
        """
        Returns the dataframe filtered by all active filters except the
        one specified.
        """
        f_df = df
        for k in filter_config:
            if k != exclude_key and st.session_state[k]:
                f_df = f_df.filter(pl.col(k).is_in(st.session_state[k]))

        # Also apply date and file filters if they are not the excluded ones
        # (These are currently treated as "always apply" for simplicity in this helper)
        if "excluded_files" in st.session_state and st.session_state.excluded_files:
            f_df = f_df.filter(
                ~pl.col("source_file").is_in(st.session_state.excluded_files)
            )

        return f_df

    # Dataset Filter
    dataset_opts = sorted(
        get_filtered_df("dataset_label")["dataset_label"].unique().to_list()
    )
    st.sidebar.multiselect("Datasets:", options=dataset_opts, key="dataset_label")

    # Model Type Filter
    model_type_opts = sorted(
        get_filtered_df("model_type")["model_type"].unique().to_list()
    )
    st.sidebar.multiselect("Model Types:", options=model_type_opts, key="model_type")

    # Metadata Filters
    with st.sidebar.expander("Algorithmic & Data Filters", expanded=True):
        # Clustering Algo Filter
        if "clustering_algo" in df.columns:
            clustering_opts = sorted(
                get_filtered_df("clustering_algo")["clustering_algo"]
                .unique()
                .drop_nulls()
                .to_list()
            )
            st.multiselect(
                "Clustering Algo:", options=clustering_opts, key="clustering_algo"
            )

        # Dim Red Algo Filter
        if "dim_red_algo" in df.columns:
            dim_red_opts = sorted(
                get_filtered_df("dim_red_algo")["dim_red_algo"]
                .unique()
                .drop_nulls()
                .to_list()
            )
            st.multiselect("Dim Red Algo:", options=dim_red_opts, key="dim_red_algo")

        # N Observations Filter
        if "n_observations" in df.columns:
            n_obs_opts = sorted(
                get_filtered_df("n_observations")["n_observations"]
                .unique()
                .drop_nulls()
                .to_list()
            )
            st.multiselect("N Observations:", options=n_obs_opts, key="n_observations")

    # Date Range Filter (Not strictly cascaded with others to avoid circular complexity,
    # but we'll use the filtered DF for available dates)
    st.sidebar.subheader("Date Filtering")
    # Use DF filtered by everything else to find valid dates
    date_filtered_df = get_filtered_df()
    valid_dates = date_filtered_df.filter(pl.col("experiment_date").is_not_null())[
        "experiment_date"
    ]

    if not valid_dates.is_empty():
        min_date, max_date = valid_dates.min(), valid_dates.max()

        date_selection = st.sidebar.date_input(
            "Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Select a start and end date for filtering experiments.",
        )

        # Handle range selection (returns a tuple of 1 or 2 items)
        if isinstance(date_selection, tuple) and len(date_selection) == 2:
            start_date, end_date = date_selection
        elif isinstance(date_selection, tuple) and len(date_selection) == 1:
            start_date = end_date = date_selection[0]
        else:
            start_date = end_date = date_selection

        # Specific Date Multiselect (Inclusion)
        all_available_dates = sorted(valid_dates.unique().to_list())
        specific_dates = st.sidebar.multiselect(
            "Filter to specific dates:",
            options=all_available_dates,
            help=(
                "If selected, only these specific dates will be shown "
                "regardless of the range above."
            ),
        )
    else:
        start_date = end_date = None
        specific_dates = []

    # Experiment Type Filter
    exp_type_opts = sorted(
        get_filtered_df("experiment_type")["experiment_type"].unique().to_list()
    )
    st.sidebar.multiselect(
        "Experiment Types:", options=exp_type_opts, key="experiment_type"
    )

    # File Exclusion Filter
    with st.sidebar.expander("Exclude Specific Files"):
        all_files = sorted(df["source_file"].unique().to_list())
        st.multiselect("Files to ignore:", options=all_files, key="excluded_files")

    # Column Visibility Selector
    st.sidebar.header("Column Visibility")
    all_columns = df.columns
    # Default columns to show (hiding more technical/verbose ones)
    default_show = [
        c for c in all_columns if c not in ["source_file", "timestamp", "dataset_name"]
    ]
    selected_columns = st.sidebar.multiselect(
        "Columns to display in table:", options=all_columns, default=default_show
    )

    # Final Filter Application
    filter_expr = pl.lit(True)  # Start with always True

    for key in filter_config:
        if st.session_state[key]:
            filter_expr = filter_expr & (pl.col(key).is_in(st.session_state[key]))

    if "excluded_files" in st.session_state and st.session_state.excluded_files:
        filter_expr = filter_expr & (
            ~pl.col("source_file").is_in(st.session_state.excluded_files)
        )

    # Date logic: use specific dates if provided, otherwise use range
    if specific_dates:
        filter_expr = filter_expr & (pl.col("experiment_date").is_in(specific_dates))
    elif start_date and end_date:
        filter_expr = filter_expr & (
            pl.col("experiment_date").is_between(start_date, end_date)
        )

    filtered_df = df.filter(filter_expr)

    if filtered_df.is_empty():
        st.info("No data matches the selected filters.")
        return

    # 3. Main Tabs
    tab_metrics, tab_qualitative, tab_coverage = st.tabs(
        [
            "📊 Quantitative Metrics",
            "🔍 Qualitative Analysis",
            "📋 Experiment Coverage",
        ]
    )

    with tab_metrics:
        # 3. Data Table with Great Tables
        st.header("📋 Consolidated Results")

        # Metadata columns list
        metadata_cols = [
            "dataset_name",
            "dataset_label",
            "experiment_date",
            "timestamp",
            "model_type",
            "model_name",
            "clustering_algo",
            "dim_red_algo",
            "experiment_type",
            "source_file",
        ]

        # Identify numeric columns for metrics
        numeric_cols = [
            col
            for col, dtype in zip(filtered_df.columns, filtered_df.dtypes)
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            and col not in metadata_cols
        ]

        # Metrics overview
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Experiments", len(filtered_df))
        m_col2.metric("Datasets", filtered_df["dataset_label"].n_unique())
        m_col3.metric("Model Types", filtered_df["model_type"].n_unique())
        avg_dur = (
            filtered_df["duration_seconds"].mean()
            if "duration_seconds" in filtered_df.columns
            else 0
        )
        m_col4.metric(
            "Avg Duration (s)",
            round(avg_dur, 2) if avg_dur is not None else 0,
        )

        # Check for NaN / None metrics in filtered_df
        nan_info = []
        for col in numeric_cols:
            null_c = filtered_df[col].null_count()
            nan_c = (
                filtered_df.filter(pl.col(col).is_nan()).shape[0]
                if filtered_df[col].dtype in [pl.Float32, pl.Float64]
                else 0
            )
            total_invalid = null_c + nan_c
            if total_invalid > 0:
                nan_info.append(f"`{col}` ({total_invalid} NaN/missing)")

        if nan_info:
            cols_str = ", ".join(nan_info)
            st.error(
                f"🚨 **Metric Errors Detected:** Found invalid values in "
                f"{len(nan_info)} column(s): {cols_str}. "
                "Cells with NaNs/errors are styled in red below."
            )

        # Table with highlighting (Reverted to Pandas Style as requested)
        # 1. Cast n_clusters to integer if it exists
        display_df = filtered_df.clone()
        if "n_clusters" in display_df.columns:
            display_df = display_df.with_columns(
                pl.col("n_clusters").cast(pl.Int64, strict=False)
            )

        import pandas as pd

        pdf = display_df.to_pandas()

        def highlight_metrics(s):
            styles = [""] * len(s)
            is_metric = s.name in METRIC_CONFIG
            is_numeric = s.name in numeric_cols

            if is_metric or is_numeric:
                numeric_s = pd.to_numeric(s, errors="coerce")

                for i, val in enumerate(s):
                    num_val = numeric_s.iloc[i]
                    if pd.isna(val) or pd.isna(num_val):
                        styles[i] = (
                            "background-color: #F8D7DA; color: #721C24; "
                            "font-weight: bold; border: 1px solid #F5C6CB;"
                        )

                if is_metric:
                    valid_s = numeric_s.dropna()
                    if not valid_s.empty:
                        direction = METRIC_CONFIG[s.name]
                        best_val = (
                            valid_s.max() if direction == "max" else valid_s.min()
                        )
                        for i, num_val in enumerate(numeric_s):
                            if (
                                pd.notna(num_val)
                                and num_val == best_val
                                and styles[i] == ""
                            ):
                                styles[i] = (
                                    "background-color: #2E7D32; "
                                    "color: white; font-weight: bold;"
                                )

            return styles

        # Filter pdf to selected columns for display, while keeping original
        # pdf for backend/plotting
        pdf_display = pdf[selected_columns] if selected_columns else pdf
        styled_table = pdf_display.style.apply(highlight_metrics).format(
            na_rep="⚠️ NaN (Error)"
        )
        st.dataframe(styled_table, width="stretch")

        # 4. Dynamic Plotting
        st.divider()
        st.header("📈 Visualization")

        pdf = filtered_df.to_pandas()
        plot_col1, plot_col2 = st.columns([1, 3])

        with plot_col1:
            st.subheader("Plot Settings")

            # Calculate default indices based on constants
            try:
                x_default_idx = numeric_cols.index(DEFAULT_X_AXIS)
            except ValueError:
                x_default_idx = 0

            try:
                y_default_idx = numeric_cols.index(DEFAULT_Y_AXIS)
            except ValueError:
                y_default_idx = min(1, len(numeric_cols) - 1)

            x_axis = st.selectbox("X-Axis", options=numeric_cols, index=x_default_idx)
            y_axis = st.selectbox("Y-Axis", options=numeric_cols, index=y_default_idx)

            color_options = [
                "model_type",
                "dataset_label",
                "experiment_date",
                "experiment_type",
            ] + numeric_cols
            color_by = st.selectbox("Color By", options=color_options, index=0)

        with plot_col2:
            is_numeric_color = color_by in numeric_cols
            color_shorthand = f"{color_by}:Q" if is_numeric_color else f"{color_by}:N"

            # In Altair, we convert date objects to ISO strings or handle them
            # as temporal
            # but for simple categorical coloring, :N works fine even with
            # date objects

            chart = (
                alt.Chart(pdf)
                .mark_circle(size=100)
                .encode(
                    x=alt.X(x_axis, scale=alt.Scale(zero=False)),
                    y=alt.Y(y_axis, scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        color_shorthand,
                        scale=alt.Scale(
                            scheme="viridis" if is_numeric_color else "tableau10"
                        ),
                    ),
                    tooltip=metadata_cols + numeric_cols,
                )
                .interactive()
                .properties(height=500)
            )
            st.altair_chart(chart, width="stretch")

    with tab_qualitative:
        st.header("🔍 Qualitative Topic Analysis")

        if qual_df.is_empty():
            st.warning("No qualitative results found in `output/`.")
        else:
            # Filter qualitative results to match the current selection
            # We use multiple criteria to be robust against file renaming/aggregation
            active_datasets = filtered_df["dataset_label"].unique().to_list()
            active_model_types = filtered_df["model_type"].unique().to_list()
            active_timestamps = []
            if "timestamp" in filtered_df.columns:
                active_timestamps = (
                    filtered_df["timestamp"].unique().drop_nulls().to_list()
                )

            # Base filter: Match by Dataset and Model Type
            filtered_qual_df = qual_df.filter(
                (pl.col("dataset_label").is_in(active_datasets))
                & (pl.col("model_type").is_in(active_model_types))
            )

            # Refinement: If we have specific timestamps for the selected runs, use them
            if active_timestamps:
                ts_filtered = filtered_qual_df.filter(
                    pl.col("timestamp").is_in(active_timestamps)
                )
                # Only use timestamp filter if it doesn't result in an empty set
                # (helps handle cases with slightly mismatched timestamps or
                # missing data)
                if not ts_filtered.is_empty():
                    filtered_qual_df = ts_filtered

            if filtered_qual_df.is_empty():
                st.info("No qualitative data matches the current filters.")
            else:
                # 1. Keyword Search
                st.subheader("🔦 Keyword Search")
                search_query = st.text_input(
                    "Search for keywords in topic representations or "
                    "representative docs:",
                    placeholder="e.g., 'covid' or 'fake news'",
                )

                if search_query:
                    search_expr = pl.col("representation").str.contains(
                        search_query, literal=False
                    ) | pl.col("representative_docs").str.contains(
                        search_query, literal=False
                    )
                    search_results = filtered_qual_df.filter(search_expr)
                    st.write(
                        f"Found {len(search_results)} topics matching '{search_query}'."
                    )
                    st.dataframe(search_results.to_pandas(), width="stretch")

                st.divider()

                # 2. Side-by-Side Comparison
                st.subheader("⚖️ Side-by-Side Model Comparison")

                # Create a selection of unique model identifiers from the
                # filtered results
                # We'll use a combination of source_file and model_id to be unique
                filtered_qual_df = filtered_qual_df.with_columns(
                    pl.concat_str(
                        [pl.col("source_file"), pl.lit(" | "), pl.col("model_id")]
                    ).alias("unique_model_id")
                )
                model_options = (
                    filtered_qual_df["unique_model_id"].unique().sort().to_list()
                )

                col_left, col_right = st.columns(2)

                with col_left:
                    model_a = st.selectbox(
                        "Select Model A:", options=model_options, index=0
                    )
                    df_a = filtered_qual_df.filter(pl.col("unique_model_id") == model_a)
                    st.write(f"**Topics for {model_a.split('|')[-1].strip()}**")
                    st.dataframe(
                        df_a.select(
                            ["topic_id", "count", "name", "representation"]
                        ).to_pandas(),
                        width="stretch",
                        hide_index=True,
                    )

                with col_right:
                    # Default to second model if available
                    default_idx = 1 if len(model_options) > 1 else 0
                    model_b = st.selectbox(
                        "Select Model B:", options=model_options, index=default_idx
                    )
                    df_b = filtered_qual_df.filter(pl.col("unique_model_id") == model_b)
                    st.write(f"**Topics for {model_b.split('|')[-1].strip()}**")
                    st.dataframe(
                        df_b.select(
                            ["topic_id", "count", "name", "representation"]
                        ).to_pandas(),
                        width="stretch",
                        hide_index=True,
                    )

                st.divider()

                # 3. Detailed Topic Explorer
                st.subheader("🗺️ Detailed Topic Explorer")
                selected_model = st.selectbox(
                    "Select a model to explore its topics in detail:",
                    options=model_options,
                )

                model_detail_df = filtered_qual_df.filter(
                    pl.col("unique_model_id") == selected_model
                )

                topic_ids = model_detail_df["topic_id"].sort().to_list()
                selected_topic = st.selectbox("Select Topic ID:", options=topic_ids)

                topic_data = model_detail_df.filter(
                    pl.col("topic_id") == selected_topic
                ).to_dicts()[0]

                det_col1, det_col2 = st.columns([1, 2])

                with det_col1:
                    st.metric("Topic ID", topic_data["topic_id"])
                    st.metric("Document Count", topic_data["count"])
                    st.write("**Representation (c-TF-IDF words):**")
                    # Try to parse as JSON list if it looks like one, otherwise
                    # just show
                    try:
                        repr_words = json.loads(topic_data["representation"])
                        st.write(", ".join(repr_words))
                    except Exception:
                        st.write(topic_data["representation"])

                with det_col2:
                    st.write("**Representative Documents:**")
                    try:
                        docs = json.loads(topic_data["representative_docs"])
                        for i, doc in enumerate(docs):
                            with st.expander(f"Document {i + 1}", expanded=(i == 0)):
                                st.write(doc)
                    except Exception:
                        st.write(topic_data["representative_docs"])

    with tab_coverage:
        st.header("📋 Experiment Execution Coverage")
        st.write(
            "Track which experiments defined in `experiments/` have been executed "
            "and saved in `results/` across representation and dataset conditions."
        )

        cov_col1, cov_col2, cov_col3, cov_col4 = st.columns([2, 2, 2, 2])

        with cov_col1:
            all_cov_datasets = ["fed", "anes", "yelp", "trump", "gadarian"]
            selected_cov_datasets = st.multiselect(
                "Filter Datasets:",
                options=all_cov_datasets,
                default=[],
                key="cov_datasets",
                help="Leave empty to show all datasets.",
            )

        with cov_col2:
            status_filter = st.selectbox(
                "Filter Status:",
                options=[
                    "All",
                    "Fully Completed (3/3)",
                    "Partially Completed (1-2/3)",
                    "Has Errors",
                    "Not Run (0/3)",
                ],
                index=0,
                key="cov_status_filter",
            )

        with cov_col3:
            cov_search = st.text_input(
                "Search Experiment:",
                placeholder="e.g. aligned_umap",
                key="cov_search",
            )

        with cov_col4:
            st.write("")
            st.write("")
            include_archived = st.checkbox(
                "Include Archived",
                value=False,
                key="cov_include_archived",
                help="Include YAML files from experiments/archive/",
            )

        # Scan experiments & build coverage matrix
        discovered_exps = scan_experiment_configs(
            exp_dir=PROJECT_ROOT / "experiments",
            include_archived=include_archived,
        )

        cov_matrix = build_coverage_matrix(discovered_exps, df)

        if cov_matrix.is_empty():
            st.info("No experiment configurations found.")
        else:
            # Apply coverage filters
            filtered_matrix = cov_matrix

            if selected_cov_datasets:
                filtered_matrix = filtered_matrix.filter(
                    pl.col("dataset_label").is_in(selected_cov_datasets)
                )

            if status_filter == "Fully Completed (3/3)":
                filtered_matrix = filtered_matrix.filter(
                    pl.col("coverage_status") == "Fully Completed"
                )
            elif status_filter == "Partially Completed (1-2/3)":
                filtered_matrix = filtered_matrix.filter(
                    pl.col("coverage_status") == "Partially Completed"
                )
            elif status_filter == "Has Errors":
                filtered_matrix = filtered_matrix.filter(
                    pl.col("coverage_status").is_in(
                        ["Has Errors", "Completed with Errors"]
                    )
                    | pl.col("keep_rep_stopwords").str.contains("Error")
                    | pl.col("remove_rep_stopwords").str.contains("Error")
                    | pl.col("stemmed").str.contains("Error")
                )
            elif status_filter == "Not Run (0/3)":
                filtered_matrix = filtered_matrix.filter(
                    pl.col("coverage_status") == "Not Run"
                )

            if cov_search:
                filtered_matrix = filtered_matrix.filter(
                    pl.col("experiment_name").str.contains(cov_search, literal=False)
                )

            # Summary Metrics
            total_exps = len(filtered_matrix)
            full_completed = len(
                filtered_matrix.filter(pl.col("coverage_status") == "Fully Completed")
            )
            total_cell_runs = (
                filtered_matrix["completed_count"].sum() if total_exps > 0 else 0
            )
            total_possible_cells = total_exps * 3
            overall_pct = (
                round((total_cell_runs / total_possible_cells) * 100, 1)
                if total_possible_cells > 0
                else 0.0
            )
            missing_cells = total_possible_cells - total_cell_runs

            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Experiments Listed", total_exps)
            s_col2.metric("Fully Completed (3/3)", full_completed)
            s_col3.metric("Coverage Rate", f"{overall_pct}%")
            s_col4.metric("Missing Condition Runs", missing_cells)

            st.divider()

            # Prepare Display Table
            display_cols = [
                "dataset_label",
                "experiment_name",
                "keep_rep_stopwords",
                "remove_rep_stopwords",
                "stemmed",
                "coverage_score",
            ]
            pdf_cov = filtered_matrix.select(display_cols).to_pandas()
            pdf_cov = pdf_cov.rename(
                columns={
                    "dataset_label": "Dataset",
                    "experiment_name": "Experiment Name",
                    "keep_rep_stopwords": "Keep Stopwords",
                    "remove_rep_stopwords": "Remove Stopwords",
                    "stemmed": "Stemmed",
                    "coverage_score": "Score",
                }
            )

            # Cell styling helper
            def style_cell(val):
                if isinstance(val, str):
                    if val.startswith("✅"):
                        return (
                            "background-color: #D4EDDA; "
                            "color: #155724; font-weight: bold;"
                        )
                    elif "Error" in val:
                        return (
                            "background-color: #F8D7DA; "
                            "color: #721C24; font-weight: bold;"
                        )
                    elif val.startswith("⚠️"):
                        return (
                            "background-color: #FFF3CD; "
                            "color: #856404; font-weight: bold;"
                        )
                    elif val.startswith("❌"):
                        return "background-color: #F8D7DA; color: #721C24;"
                return ""

            style_fn = getattr(pdf_cov.style, "map", None) or getattr(
                pdf_cov.style, "applymap"
            )
            styled_cov_df = style_fn(style_cell)
            st.dataframe(styled_cov_df, width="stretch", hide_index=True)

            # Detail Expander
            st.divider()
            st.subheader("🔍 Experiment Details Inspector")
            exp_names = filtered_matrix["experiment_name"].to_list()
            if exp_names:
                selected_exp_name = st.selectbox(
                    "Select Experiment to Inspect Details:", options=exp_names
                )
                exp_detail_row = filtered_matrix.filter(
                    pl.col("experiment_name") == selected_exp_name
                ).to_dicts()[0]

                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.write("**Configuration Files:**")
                    std_path = exp_detail_row.get("yaml_standard") or "N/A"
                    stem_path = exp_detail_row.get("yaml_stemmed") or "N/A"
                    st.write(f"- Standard YAML: `{std_path}`")
                    st.write(f"- Stemmed YAML: `{stem_path}`")
                    st.write(f"- Dataset: `{exp_detail_row['dataset_label']}`")
                    st.write(
                        f"- Is Archived: `{exp_detail_row.get('is_archived', False)}`"
                    )

                with d_col2:
                    st.write("**Condition Details:**")
                    run_details = json.loads(
                        exp_detail_row.get("run_details_json", "{}")
                    )
                    for cond_key, cond_title in [
                        ("keep_rep_stopwords", "Keep Stopwords"),
                        ("remove_rep_stopwords", "Remove Stopwords"),
                        ("stemmed", "Stemmed"),
                    ]:
                        c_info = run_details.get(cond_key, {})
                        status = c_info.get("status", "Not Run")
                        cnt = c_info.get("count", 0)
                        err_cnt = c_info.get("error_count", 0)
                        dry_cnt = c_info.get("dry_run_count", 0)
                        if err_cnt > 0:
                            nan_list = ", ".join(c_info.get("nan_metrics", []))
                            st.write(
                                f"- **{cond_title}**: {status} "
                                f"({cnt} valid runs, :red[{err_cnt} with NaNs/Errors], "
                                f"{dry_cnt} dry runs)"
                            )
                            if nan_list:
                                st.write(f"  - *Metrics with NaNs*: `{nan_list}`")
                        else:
                            st.write(
                                f"- **{cond_title}**: {status} "
                                f"({cnt} full runs, {dry_cnt} dry runs)"
                            )


if __name__ == "__main__":
    main()
