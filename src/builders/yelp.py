# -*- coding: utf-8 -*-
"""Yelp dataset builder."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("builder.yelp")

RELEVANT_COLS = [
    "date",
    "stars",
    "text",
    "review_count",
    "average_stars",
    "yelping_since",
    "state",
    "stars_business",
    "review_count_business",
]


def write_parquet_files(
    data_dir: Path,
    filename_prefix: str = "yelp_academic_dataset_",
    files: list[str] = ["business", "review", "user"],
):
    """
    Converts the original JSON files into parquet files for faster reading and processing.
    Datetime columns are parsed as such.
    No other changes are made to the files.
    """
    for file in files:
        filename = filename_prefix + file
        json_path = data_dir / (filename + ".json")
        parquet_path = data_dir / (filename + ".parquet")

        if not json_path.exists():
            logger.warning(
                f"JSON file not found: {json_path}. Skipping conversion for this file."
            )
            continue

        logger.info(f"Converting {json_path} to {parquet_path}...")
        df = pl.read_ndjson(json_path)

        if file == "review":
            df = df.with_columns(pl.col("date").str.to_datetime())
        elif file == "user":
            df = df.with_columns(pl.col("yelping_since").str.to_datetime())

        df.write_parquet(parquet_path)
        del df


def change_lf(lf: pl.LazyFrame, select_cols: list[str]) -> pl.LazyFrame:
    """
    Makes changes to the LazyFrame such as selecting only relevant columns
    and renaming others.
    """
    return lf.select(select_cols).rename(
        {
            "review_count_business": "business_review_count",
            "stars_business": "business_stars",
            "average_stars": "user_average_stars",
            "review_count": "user_review_count",
        }
    )


def build(raw_dir: Path, interim_dir: Path, skip_convert: bool = False):
    """Processes Yelp JSON files into a single Parquet file."""
    interim_dir.mkdir(parents=True, exist_ok=True)

    if not skip_convert:
        write_parquet_files(data_dir=raw_dir)

    business_path = raw_dir / "yelp_academic_dataset_business.parquet"
    review_path = raw_dir / "yelp_academic_dataset_review.parquet"
    user_path = raw_dir / "yelp_academic_dataset_user.parquet"

    if not all(p.exists() for p in [business_path, review_path, user_path]):
        raise FileNotFoundError(
            "One or more required Parquet files are missing in raw_dir."
        )

    business_lf = pl.scan_parquet(business_path)
    review_lf = pl.scan_parquet(review_path)
    user_lf = pl.scan_parquet(user_path)

    logger.info("Joining review, user, and business dataframes...")
    tmp_lf = review_lf.join(other=user_lf, on="user_id", suffix="_user")
    full_df = tmp_lf.join(other=business_lf, on="business_id", suffix="_business")

    full_yelp_path = raw_dir / "full_yelp_reviews.parquet"
    logger.info(f"Sinking full joined dataset to {full_yelp_path}...")
    full_df.sink_parquet(full_yelp_path)

    lf = pl.scan_parquet(full_yelp_path)
    lf = change_lf(lf, select_cols=RELEVANT_COLS)
    lf = lf.with_row_index()

    output_path = interim_dir / "yelp_reviews.parquet"
    logger.info(f"Sinking final processed dataset to {output_path}...")
    lf.sink_parquet(output_path)

    logger.info(f"Yelp dataset built and saved to {output_path}")
