"""Interactive dashboard for BERTopic experiment results analysis.

This module provides a Streamlit-based dashboard to load, filter, and visualize
experimental results from CSV files in the results directory.
"""

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import altair as alt
import polars as pl
import streamlit as st


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
}


def extract_model_type(name: str) -> str:
    """Extracts the base model type from a model name (e.g., baseline_1 -> baseline)."""
    if not name or not isinstance(name, str):
        return "unknown"
    # Handles common patterns like 'baseline_1' or 'mv_spectral_2'
    # We take the part before the last underscore if it's followed by a digit
    parts = name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return name


import datetime
import glob
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import altair as alt
import polars as pl
import streamlit as st


# ... (METRIC_CONFIG and extract_model_type remain the same)

@st.cache_data
def load_all_results(results_dir: str = "results") -> pl.DataFrame:
    """Loads and concatenates result files. Extracts dataset and date objects."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        return pl.DataFrame()

    dfs = []
    for file in csv_files:
        try:
            df = pl.read_csv(file)
            file_basename = os.path.basename(file)
            
            # Extract Date as actual date object
            date_match = re.search(r"-(\d{8})-", file_basename)
            exp_date = None
            if date_match:
                d_str = date_match.group(1)
                try:
                    exp_date = datetime.date(int(d_str[:4]), int(d_str[4:6]), int(d_str[6:]))
                except ValueError:
                    exp_date = None
            
            # Dataset extraction
            dataset = "unknown"
            if "trump" in file_basename.lower():
                dataset = "trump"
            elif "yelp" in file_basename.lower():
                dataset = "yelp"
            else:
                dataset = file_basename.split("-")[0].split("_")[0]

            df = df.with_columns(
                pl.lit(file_basename).alias("source_file"),
                pl.lit(dataset).alias("dataset"),
                pl.lit(exp_date).cast(pl.Date).alias("experiment_date"),
                pl.lit("optimizer" if "opt" in file_basename.lower() else "non-optimizer").alias(
                    "experiment_type"
                ),
            )
            
            if "model_name" in df.columns:
                df = df.with_columns(
                    pl.col("model_name").map_elements(extract_model_type, return_dtype=pl.String).alias("model_type")
                )
            else:
                df = df.with_columns(pl.lit("unknown").alias("model_type"))
                
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="diagonal")


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
    df = load_all_results(results_dir)

    if df.is_empty():
        st.warning(f"No result files found in `{results_dir}/`.")
        return

    # 2. Sidebar Filters
    st.sidebar.header("Data Filters")

    # Dataset Filter
    all_datasets = sorted(df["dataset"].unique().to_list())
    selected_datasets = st.sidebar.multiselect("Datasets:", options=all_datasets, default=all_datasets)

    # Model Type Filter
    all_model_types = sorted(df["model_type"].unique().to_list())
    selected_model_types = st.sidebar.multiselect("Model Types:", options=all_model_types, default=all_model_types)

    # Date Range Filter
    st.sidebar.subheader("Date Filtering")
    valid_dates = df.filter(pl.col("experiment_date").is_not_null())["experiment_date"]
    
    if not valid_dates.is_empty():
        min_date, max_date = valid_dates.min(), valid_dates.max()
        
        date_selection = st.sidebar.date_input(
            "Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Select a start and end date for filtering experiments."
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
            help="If selected, only these specific dates will be shown regardless of the range above."
        )
    else:
        start_date = end_date = None
        specific_dates = []

    # Experiment Type Filter
    exp_types = df["experiment_type"].unique().to_list()
    selected_types = st.sidebar.multiselect("Experiment Types:", options=exp_types, default=exp_types)

    # File Exclusion Filter
    with st.sidebar.expander("Exclude Specific Files"):
        all_files = sorted(df["source_file"].unique().to_list())
        excluded_files = st.multiselect("Files to ignore:", options=all_files)

    # Apply all filters
    filter_expr = (
        (pl.col("dataset").is_in(selected_datasets)) &
        (pl.col("model_type").is_in(selected_model_types)) &
        (pl.col("experiment_type").is_in(selected_types)) &
        (~pl.col("source_file").is_in(excluded_files))
    )
    
    # Date logic: use specific dates if provided, otherwise use range
    if specific_dates:
        filter_expr = filter_expr & (pl.col("experiment_date").is_in(specific_dates))
    elif start_date and end_date:
        filter_expr = filter_expr & (pl.col("experiment_date").is_between(start_date, end_date))

    filtered_df = df.filter(filter_expr)

    if filtered_df.is_empty():
        st.info("No data matches the selected filters.")
        return

    # 3. Data Table with Highlighting
    st.header("📋 Consolidated Results")

    # Metadata columns list
    metadata_cols = ["dataset", "experiment_date", "model_type", "model_name", "experiment_type", "source_file"]
    
    # Identify numeric columns for metrics
    numeric_cols = [
        col for col, dtype in zip(filtered_df.columns, filtered_df.dtypes)
        if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        and col not in metadata_cols
    ]

    # Metrics overview
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Experiments", len(filtered_df))
    m_col2.metric("Datasets", filtered_df["dataset"].n_unique())
    m_col3.metric("Model Types", filtered_df["model_type"].n_unique())
    m_col4.metric("Avg Duration (s)", round(filtered_df["duration_seconds"].mean(), 2) if "duration_seconds" in filtered_df.columns else 0)

    # Table with highlighting
    pdf = filtered_df.to_pandas()
    def highlight_best(s):
        if s.name in METRIC_CONFIG:
            direction = METRIC_CONFIG[s.name]
            is_best = (s == s.max()) if direction == "max" else (s == s.min())
            return ["background-color: #2E7D32; color: white" if v else "" for v in is_best]
        return [""] * len(s)

    st.dataframe(pdf.style.apply(highlight_best), use_container_width=True)

    # 4. Dynamic Plotting
    st.divider()
    st.header("📈 Visualization")

    plot_col1, plot_col2 = st.columns([1, 3])

    with plot_col1:
        st.subheader("Plot Settings")
        x_axis = st.selectbox("X-Axis", options=numeric_cols, index=0)
        y_axis = st.selectbox("Y-Axis", options=numeric_cols, index=min(1, len(numeric_cols)-1))
        
        color_options = ["model_type", "dataset", "experiment_date", "experiment_type"] + numeric_cols
        color_by = st.selectbox("Color By", options=color_options, index=0)

    with plot_col2:
        is_numeric_color = color_by in numeric_cols
        color_shorthand = f"{color_by}:Q" if is_numeric_color else f"{color_by}:N"
        
        # In Altair, we convert date objects to ISO strings or handle them as temporal
        # but for simple categorical coloring, :N works fine even with date objects
        
        chart = (
            alt.Chart(pdf).mark_circle(size=100).encode(
                x=alt.X(x_axis, scale=alt.Scale(zero=False)),
                y=alt.Y(y_axis, scale=alt.Scale(zero=False)),
                color=alt.Color(color_shorthand, scale=alt.Scale(scheme="viridis" if is_numeric_color else "tableau10")),
                tooltip=metadata_cols + numeric_cols
            ).interactive().properties(height=500)
        )
        st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
