# -*- coding: utf-8 -*-
"""FED dataset builder."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("builder.fed")


def load_macro_data(
    raw_dir: Path, file_name: str, value_col: str, new_name: str
) -> pl.DataFrame:
    """Loads a macro indicator CSV, parses dates, and includes both unlagged and 1-observation lagged values."""
    path = raw_dir / file_name
    df = pl.read_csv(path)
    # Ensure date is parsed and sorted for join_asof
    df = df.with_columns(pl.col("observation_date").str.to_date())
    df = df.sort("observation_date")

    # Fill missing values using a local window average (rolling mean of 5: 2 before, 2 after)
    # This addresses gaps like those from government shutdowns using nearby values.
    df = df.with_columns(
        pl.col(value_col)
        .fill_null(
            pl.col(value_col).rolling_mean(window_size=5, center=True, min_samples=1)
        )
        .forward_fill()
        .backward_fill()
        .alias(value_col)
    )

    # Create both unlagged and lagged versions with simplified names
    df = df.with_columns(
        [
            pl.col(value_col).alias(new_name),
            pl.col(value_col).shift(1).alias(f"{new_name}_lag"),
        ]
    )
    # Handle the first row's lag null
    df = df.with_columns(pl.col(f"{new_name}_lag").forward_fill().backward_fill())

    return df.select(["observation_date", new_name, f"{new_name}_lag"])


def build(raw_dir: Path, interim_dir: Path):
    """Builds the FED dataset by joining communications with macro indicators and political metadata."""
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Communications (The base)
    comm_df = pl.read_csv(raw_dir / "communications.csv")
    comm_df = (
        comm_df.with_columns(
            pl.col("Type").alias("type"),
            pl.col("Date").str.to_date().alias("date"),
            pl.col("Release Date").str.to_date().alias("release_date"),
            # Collapse all whitespace (tabs, newlines, multiple spaces) into single spaces
            pl.col("Text").str.replace_all(r"\s+", " ").str.strip_chars().alias("text"),
        )
        .drop(["Date", "Release Date", "Text", "Type"])
        .sort("date")
    )

    # 2. Load Macro Indicators (Using 1999_2026 series with monthly and yearly variants)
    gdp_m_df = load_macro_data(
        raw_dir, "us_gdp_monthly_1999_2026.csv", "GDP_PCH", "gdp_monthly"
    )
    gdp_y_df = load_macro_data(
        raw_dir, "us_gdp_yearly_1999_2026.csv", "GDP_PC1", "gdp_yearly"
    )
    cpi_m_df = load_macro_data(
        raw_dir, "cpi_monthly_1999_2026.csv", "CPIAUCSL_PCH", "cpi_monthly"
    )
    cpi_y_df = load_macro_data(
        raw_dir, "cpi_yearly_1999_2026.csv", "CPIAUCSL_PC1", "cpi_yearly"
    )
    funds_df = load_macro_data(
        raw_dir, "fed_funds_1999_2026.csv", "FEDFUNDS", "funds_rate"
    )
    unrate_df = load_macro_data(
        raw_dir, "unemployment_1999_2026.csv", "UNRATE", "unemployment"
    )

    # 3. Load Political Metadata (Daily series)
    pol_df = pl.read_csv(raw_dir / "presidents_and_chairmen.csv")
    pol_df = pol_df.with_columns(pl.col("date").str.to_date()).sort("date")

    # 4. Sequential Joins using join_asof (backward)
    fed_df = comm_df.join_asof(
        gdp_m_df, left_on="date", right_on="observation_date", strategy="backward"
    ).drop("observation_date")

    fed_df = fed_df.join_asof(
        gdp_y_df, left_on="date", right_on="observation_date", strategy="backward"
    ).drop("observation_date")

    fed_df = fed_df.join_asof(
        cpi_m_df, left_on="date", right_on="observation_date", strategy="backward"
    ).drop("observation_date")

    fed_df = fed_df.join_asof(
        cpi_y_df, left_on="date", right_on="observation_date", strategy="backward"
    ).drop("observation_date")

    fed_df = fed_df.join_asof(
        funds_df, left_on="date", right_on="observation_date", strategy="backward"
    ).drop("observation_date")

    fed_df = fed_df.join_asof(
        unrate_df, left_on="date", right_on="observation_date", strategy="backward"
    ).drop("observation_date")

    fed_df = fed_df.join_asof(pol_df, on="date", strategy="backward")

    # 5. Final Cleanup and Save
    # Handle missing release dates: same day for statements, +21 days for minutes
    fed_df = fed_df.with_columns(
        pl.when(pl.col("release_date").is_null())
        .then(
            pl.when(pl.col("type") == "Statement")
            .then(pl.col("date"))
            .otherwise(pl.col("date").dt.offset_by("21d"))
        )
        .otherwise(pl.col("release_date"))
        .alias("release_date")
    )

    output_path = interim_dir / "fed_communications.parquet"
    fed_df.write_parquet(output_path)
    logger.info(f"FED dataset built and saved to {output_path}")
    logger.info(f"Columns: {fed_df.columns}")
    logger.info(f"Total rows: {len(fed_df)}")
