# -*- coding: utf-8 -*-
"""Gadarian dataset builder."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("builder.gadarian")


def build(raw_dir: Path, interim_dir: Path):
    """Builds the Gadarian dataset by loading the raw CSV and applying
    transformations.
    """
    interim_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "gadarian.csv"
    if not raw_path.exists():
        logger.error(f"Raw dataset not found: {raw_path}")
        return

    logger.info("Loading Gadarian dataset...")
    df = pl.read_csv(raw_path)

    logger.info("Transforming Gadarian dataset...")

    # Define the mapping for partisanship based on the 0-1 scale
    # (0=Strong Rep, 1=Strong Dem)
    def map_partisanship(val):
        if val is None:
            return "Unknown"
        # Multiplying by 6 to get back to the 0-6 integer scale
        int_val = round(val * 6)
        mapping = {
            0: "Strong Republican",
            1: "Republican",
            2: "Weak Republican",
            3: "Independent",
            4: "Weak Democrat",
            5: "Democrat",
            6: "Strong Democrat",
        }
        return mapping.get(int_val, "Unknown")

    df = (
        df.drop("MetaID")
        .rename({"pid_rep": "partisanship", "open.ended.response": "text"})
        .with_row_index(name="index")
        .with_columns(
            pl.col("treatment").cast(pl.Boolean),
            pl.col("partisanship")
            .map_elements(map_partisanship, return_dtype=pl.Utf8)
            .cast(pl.Categorical),
        )
    )

    output_path = interim_dir / "gadarian.parquet"
    df.write_parquet(output_path)
    logger.info(f"Gadarian dataset built and saved to {output_path}")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Total rows: {len(df)}")
