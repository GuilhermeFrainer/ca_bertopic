# -*- coding: utf-8 -*-
"""Unified script to build different datasets for topic modeling."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.builders import anes, fed, gadarian, trump, yelp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_datasets")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def main():
    parser = argparse.ArgumentParser(description="Build datasets for topic modeling.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "fed",
            "yelp",
            "anes",
            "gadarian",
            "trump",
        ],  # Add future datasets here
        help="The dataset to build.",
    )

    # Yelp-specific arguments
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="[Yelp only] Skip the JSON-to-Parquet conversion step.",
    )

    args = parser.parse_args()

    if args.dataset == "fed":
        logger.info("Building FED dataset...")
        fed.build(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR)
    elif args.dataset == "yelp":
        logger.info("Building Yelp dataset...")
        yelp.build(
            raw_dir=RAW_DIR, interim_dir=INTERIM_DIR, skip_convert=args.skip_convert
        )
    elif args.dataset == "anes":
        logger.info("Building ANES dataset...")
        anes.build(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR)
    elif args.dataset == "gadarian":
        logger.info("Building Gadarian dataset...")
        gadarian.build(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR)
    elif args.dataset == "trump":
        logger.info("Building Trump dataset...")
        trump.build(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR)
    # Add future datasets here:
    # elif args.dataset == "new_dataset":
    #     logger.info("Building New Dataset...")
    #     new_dataset_builder.build(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR)


if __name__ == "__main__":
    main()
