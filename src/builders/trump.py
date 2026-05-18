# -*- coding: utf-8 -*-
"""Trump tweets dataset builder."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("builder.trump")


def build(raw_dir: Path, interim_dir: Path):
    """Builds the Trump tweets dataset by loading the raw CSV and applying
    basic transformations.
    """
    interim_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "trump_tweets.csv"
    if not raw_path.exists():
        logger.error(f"Raw dataset not found: {raw_path}")
        return

    logger.info("Loading Trump tweets dataset...")
    # Using ignore_errors=True for robustness with some potentially malformed rows
    df = pl.read_csv(raw_path, ignore_errors=True)

    logger.info("Transforming Trump tweets dataset...")

    df = (
        df.with_columns(
            # Convert boolean-like columns ('t'/'f') to actual booleans
            (pl.col("isRetweet").str.to_lowercase() == "t").alias("is_retweet"),
            (pl.col("isDeleted").str.to_lowercase() == "t").alias("is_deleted"),
            (pl.col("isFlagged").str.to_lowercase() == "t").alias("is_flagged"),
            # Convert date column
            pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            # Cast device to categorical
            pl.col("device").cast(pl.Categorical),
        )
        .drop(["isRetweet", "isDeleted", "isFlagged"])
    )

    # Ensure all other columns are snake_case if they aren't already
    df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns}).with_row_index()

    output_path = interim_dir / "trump.parquet"
    df.write_parquet(output_path)
    logger.info(f"Trump dataset built and saved to {output_path}")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Total rows: {len(df)}")
