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
import os
import re
from typing import List, Optional, Tuple

import altair as alt
import polars as pl
import streamlit as st

import src.make_table as make_table

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
    # Handles common patterns like 'baseline_1' or 'mv_spectral_2'
    # We take the part before the last underscore if it's followed by a digit
    parts = name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return name


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
            
            # Dataset extraction (New logic: use dataset_name column if it exists)
            dataset = "unknown"
            if "dataset_name" in df.columns:
                dataset = df["dataset_name"][0]
            elif "trump" in file_basename.lower():
                dataset = "trump"
            elif "yelp" in file_basename.lower():
                dataset = "yelp"
            else:
                dataset = file_basename.split("-")[0].split("_")[0]

            df = df.with_columns(
                pl.lit(file_basename).alias("source_file"),
                pl.lit(dataset).alias("dataset_label"),
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
    all_datasets = sorted(df["dataset_label"].unique().to_list())
    selected_datasets = st.sidebar.multiselect("Datasets:", options=all_datasets, default=all_datasets)

    # Model Type Filter
    all_model_types = sorted(df["model_type"].unique().to_list())
    selected_model_types = st.sidebar.multiselect("Model Types:", options=all_model_types, default=all_model_types)

    # Metadata Filters
    with st.sidebar.expander("Algorithmic & Data Filters", expanded=True):
        # Clustering Algo Filter
        if "clustering_algo" in df.columns:
            all_clustering = sorted(df["clustering_algo"].unique().drop_nulls().to_list())
            selected_clustering = st.multiselect("Clustering Algo:", options=all_clustering, default=all_clustering)
        else:
            selected_clustering = None

        # Dim Red Algo Filter
        if "dim_red_algo" in df.columns:
            all_dim_red = sorted(df["dim_red_algo"].unique().drop_nulls().to_list())
            selected_dim_red = st.multiselect("Dim Red Algo:", options=all_dim_red, default=all_dim_red)
        else:
            selected_dim_red = None

        # N Observations Filter
        if "n_observations" in df.columns:
            all_n_obs = sorted(df["n_observations"].unique().drop_nulls().to_list())
            selected_n_obs = st.multiselect("N Observations:", options=all_n_obs, default=all_n_obs)
        else:
            selected_n_obs = None

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

    # Column Visibility Selector
    st.sidebar.header("Column Visibility")
    all_columns = df.columns
    # Default columns to show (hiding more technical/verbose ones)
    default_show = [
        c for c in all_columns 
        if c not in ["source_file", "timestamp", "dataset_name"]
    ]
    selected_columns = st.sidebar.multiselect(
        "Columns to display in table:", 
        options=all_columns, 
        default=default_show
    )

    # Apply all filters
    filter_expr = (
        (pl.col("dataset_label").is_in(selected_datasets)) &
        (pl.col("model_type").is_in(selected_model_types)) &
        (pl.col("experiment_type").is_in(selected_types)) &
        (~pl.col("source_file").is_in(excluded_files))
    )
    
    if selected_clustering is not None:
        filter_expr = filter_expr & (pl.col("clustering_algo").is_in(selected_clustering) | pl.col("clustering_algo").is_null())
    if selected_dim_red is not None:
        filter_expr = filter_expr & (pl.col("dim_red_algo").is_in(selected_dim_red) | pl.col("dim_red_algo").is_null())
    if selected_n_obs is not None:
        filter_expr = filter_expr & (pl.col("n_observations").is_in(selected_n_obs) | pl.col("n_observations").is_null())

    # Date logic: use specific dates if provided, otherwise use range
    if specific_dates:
        filter_expr = filter_expr & (pl.col("experiment_date").is_in(specific_dates))
    elif start_date and end_date:
        filter_expr = filter_expr & (pl.col("experiment_date").is_between(start_date, end_date))

    filtered_df = df.filter(filter_expr)

    if filtered_df.is_empty():
        st.info("No data matches the selected filters.")
        return

    # 3. Data Table with Great Tables
    st.header("📋 Consolidated Results")

    # Metadata columns list
    metadata_cols = [
        "dataset_name", "dataset_label", "experiment_date", "timestamp",
        "model_type", "model_name", "clustering_algo", "dim_red_algo",
        "experiment_type", "source_file"
    ]
    
    # Identify numeric columns for metrics
    numeric_cols = [
        col for col, dtype in zip(filtered_df.columns, filtered_df.dtypes)
        if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        and col not in metadata_cols
    ]

    # Metrics overview
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Experiments", len(filtered_df))
    m_col2.metric("Datasets", filtered_df["dataset_label"].n_unique())
    m_col3.metric("Model Types", filtered_df["model_type"].n_unique())
    m_col4.metric("Avg Duration (s)", round(filtered_df["duration_seconds"].mean(), 2) if "duration_seconds" in filtered_df.columns else 0)

    # Table with highlighting (Reverted to Pandas Style as requested)
    # 1. Cast n_clusters to integer if it exists
    display_df = filtered_df.clone()
    if "n_clusters" in display_df.columns:
        display_df = display_df.with_columns(pl.col("n_clusters").cast(pl.Int64, strict=False))
    
    pdf = display_df.to_pandas()
    
    def highlight_best(s):
        if s.name in METRIC_CONFIG:
            direction = METRIC_CONFIG[s.name]
            is_best = (s == s.max()) if direction == "max" else (s == s.min())
            return ["background-color: #2E7D32; color: white" if v else "" for v in is_best]
        return [""] * len(s)

    # Filter pdf to selected columns for display, while keeping original pdf for backend/plotting
    pdf_display = pdf[selected_columns] if selected_columns else pdf
    st.dataframe(pdf_display.style.apply(highlight_best), use_container_width=True)

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
            y_default_idx = min(1, len(numeric_cols)-1)

        x_axis = st.selectbox("X-Axis", options=numeric_cols, index=x_default_idx)
        y_axis = st.selectbox("Y-Axis", options=numeric_cols, index=y_default_idx)
        
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
