# -*- coding: utf-8 -*-
"""ANES 2008 dataset builder."""

import logging
from pathlib import Path

import pandas as pd
import polars as pl

logger = logging.getLogger("builder.anes")


def build(raw_dir: Path, interim_dir: Path):
    """Builds the ANES 2008 dataset by joining open-ended responses with metadata."""
    interim_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = raw_dir / "anes_timeseries_2008.dta"
    text_responses_path = (
        raw_dir / "anes_timeseries_2008_openends_redacted_Dec2012Revision.xls"
    )

    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return
    if not text_responses_path.exists():
        logger.error(f"Text responses file not found: {text_responses_path}")
        return

    logger.info("Loading ANES metadata...")
    relevant_metadata_columns = [
        "V080001",  # Case ID
        "V083097",  # Party identification
        "V083217",  # Years of education
        "V083215x",  # Age
    ]

    # Use pandas to read .dta as polars doesn't support it directly
    pd_df = pd.read_stata(metadata_path)
    pd_df = pd_df[relevant_metadata_columns]
    for column in relevant_metadata_columns:
        pd_df[column] = pd_df[column].astype("string")

    metadata_df = (
        pl.from_pandas(pd_df)
        .rename(
            {
                "V080001": "id",
                "V083097": "party_id",
                "V083217": "years_of_education",
                "V083215x": "age",
            }
        )
        .with_columns(
            pl.col("id").cast(pl.Float64).cast(pl.Int64),
            pl.when(pl.col("age").is_in(["-9. Refused", "-8. Don't know"]))
            .then(None)
            .otherwise(pl.col("age"))
            .alias("age"),
            pl.when(
                pl.col("years_of_education").is_in(["-9. Refused", "-8. Don't know"])
            )
            .then(None)
            .otherwise(pl.col("years_of_education"))
            .alias("years_of_education"),
        )
        .with_columns(
            pl.col("age").cast(pl.Float64).cast(pl.Int64),
            pl.col("years_of_education").cast(pl.Float64).cast(pl.Int64),
        )
    )

    logger.info("Loading ANES text responses...")
    text_df = (
        pl.read_excel(text_responses_path, sheet_name="MIPpolit1")[1:]
        .rename(
            {
                "caseID": "id",
                "Q3b1. CSES_ISSPOLITICAL1 (CSES Module: Most Important "
                "Political Issue)": "text",
            }
        )
        .drop("post-election IW")
        .with_columns(pl.col("id").cast(pl.Int64))
    )

    logger.info("Merging and transforming ANES dataset...")
    party_map = {
        "1. Democrat": "Democrat",
        "2. Republican": "Republican",
        "3. Independent": "Independent",
        "4. Other party (SPECIFY)": "Other",
        "5. No preference {VOL}": "No Preference",
        "-8. Don't know": "Unknown",
        "-9. Refused": "Refused",
    }

    anes_df = (
        text_df.join(metadata_df, on="id")
        .with_columns(pl.col("party_id").replace(party_map).cast(pl.Categorical))
        .drop_nulls()
    )

    output_path = interim_dir / "anes_2008.parquet"
    anes_df.write_parquet(output_path)
    logger.info(f"ANES dataset built and saved to {output_path}")
    logger.info(f"Columns: {anes_df.columns}")
    logger.info(f"Total rows: {len(anes_df)}")
