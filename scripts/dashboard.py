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


def load_all_results(results_dir: str = "results") -> pl.DataFrame:
    """Loads and concatenates all CSV result files from the results directory.

    Args:
        results_dir: Path to the directory containing CSV result files.

    Returns:
        A Polars DataFrame containing consolidated results with a 'source_file' column.
    """
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        return pl.DataFrame()

    dfs = []
    for file in csv_files:
        try:
            # Using Polars to read the CSV
            df = pl.read_csv(file)
            # Add metadata about the source file
            df = df.with_columns(
                pl.lit(os.path.basename(file)).alias("source_file"),
                pl.lit("optimizer" if "opt" in file.lower() else "non-optimizer").alias(
                    "experiment_type"
                ),
            )
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    if not dfs:
        return pl.DataFrame()

    # Vertical concatenation. We use how="diagonal" because optimizer results
    # might have extra columns for hyperparameters.
    return pl.concat(dfs, how="diagonal")


def main():
    """Main function to run the Streamlit dashboard."""
    st.set_page_config(
        page_title="CA-BERTopic Experiment Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 CA-BERTopic Experiment Dashboard")
    st.markdown(
        "Analyze and compare topic modeling experiments and optimizer results."
    )

    # 1. Load Data
    results_dir = "results"
    df = load_all_results(results_dir)

    if df.is_empty():
        st.warning(f"No result files found in `{results_dir}/`.")
        return

    # 2. Sidebar Filters
    st.sidebar.header("Filters & Options")

    # File Exclusion Filter
    all_files = sorted(df["source_file"].unique().to_list())
    excluded_files = st.sidebar.multiselect(
        "Exclude specific files:",
        options=all_files,
        help="Select result files to ignore in the analysis.",
    )

    # Filter by Experiment Type
    exp_types = df["experiment_type"].unique().to_list()
    selected_types = st.sidebar.multiselect(
        "Filter by experiment type:",
        options=exp_types,
        default=exp_types,
    )

    # Filter the dataframe
    filtered_df = df.filter(
        (~pl.col("source_file").is_in(excluded_files))
        & (pl.col("experiment_type").is_in(selected_types))
    )

    if filtered_df.is_empty():
        st.info("No data matches the selected filters.")
        return

    # 3. Data Table with Highlighting
    st.header("📋 Consolidated Results")

    # Identify numeric columns for metrics and highlighting
    # We exclude columns that are identifiers or metadata
    metadata_cols = ["source_file", "experiment_type", "model_name"]
    numeric_cols = [
        col
        for col, dtype in zip(filtered_df.columns, filtered_df.dtypes)
        if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        and col not in metadata_cols
    ]

    # Display basic stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Experiments", len(filtered_df))
    col2.metric("Result Files", filtered_df["source_file"].n_unique())
    col3.metric("Available Metrics", len(numeric_cols))

    # Function to highlight "best" (max value) in each column
    # Streamlit's st.dataframe supports Pandas-style styling.
    # We convert to pandas for styling as Polars doesn't have direct styling support yet.
    pdf = filtered_df.to_pandas()

    def highlight_max(s):
        """Highlights the maximum value in a Series."""
        if s.name in numeric_cols:
            is_max = s == s.max()
            return ["background-color: #2E7D32; color: white" if v else "" for v in is_max]
        return [""] * len(s)

    styled_df = pdf.style.apply(highlight_max)
    st.dataframe(styled_df, use_container_width=True)

    # 4. Dynamic Plotting
    st.divider()
    st.header("📈 Visualization")

    plot_col1, plot_col2 = st.columns([1, 3])

    with plot_col1:
        st.subheader("Plot Settings")
        x_axis = st.selectbox("X-Axis Metric", options=numeric_cols, index=0)
        y_axis = st.selectbox(
            "Y-Axis Metric",
            options=numeric_cols,
            index=min(1, len(numeric_cols) - 1),
        )
        color_by = st.selectbox(
            "Color By", options=["experiment_type", "model_name", "source_file"]
        )

    with plot_col2:
        # Create Altair chart
        chart = (
            alt.Chart(pdf)
            .mark_circle(size=100)
            .encode(
                x=alt.X(x_axis, scale=alt.Scale(zero=False)),
                y=alt.Y(y_axis, scale=alt.Scale(zero=False)),
                color=color_by,
                tooltip=metadata_cols + numeric_cols,
            )
            .interactive()
            .properties(height=500)
        )
        st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
