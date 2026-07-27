# -*- coding: utf-8 -*-
"""Samples 10,000 documents from interim Yelp parquet data upfront."""

import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.logger_config as logger_config
from src.data import sample_from_lf

RAW_YELP = PROJECT_ROOT / "data/interim/yelp_reviews.parquet"
OUTPUT_YELP_S10000 = PROJECT_ROOT / "data/interim/yelp_s10000_raw.parquet"
SAMPLE_SIZE = 10000
RANDOM_STATE = 36201624


def main():
    logger = logger_config.setup_logging("sample_yelp_interim", PROJECT_ROOT / "logs")
    logger.info("Starting early Yelp sampling...")

    if not RAW_YELP.exists():
        logger.error(f"Input file not found: {RAW_YELP}. Build yelp dataset first.")
        sys.exit(1)

    logger.info(f"Scanning {RAW_YELP}...")
    lf = pl.scan_parquet(RAW_YELP)

    logger.info(f"Sampling {SAMPLE_SIZE} reviews with seed {RANDOM_STATE}...")
    sampled_df = sample_from_lf(lf, n=SAMPLE_SIZE, seed=RANDOM_STATE).collect()

    logger.info(f"Saving sampled raw Yelp to {OUTPUT_YELP_S10000}...")
    sampled_df.write_parquet(OUTPUT_YELP_S10000)
    logger.info("Early Yelp sampling completed successfully.")


if __name__ == "__main__":
    main()
