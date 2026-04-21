# -*- coding: utf-8 -*-
"""Preprocesses the Trump and Yelp datasets for topic modeling.

This script executes the preprocessing logic defined in src/processing.py.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processing import process_dataset

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATHS = {
    "trump": PROJECT_ROOT / "data/raw/trump_tweets.csv",
    "yelp": PROJECT_ROOT / "data/interim/yelp_reviews.parquet",
    "fed": PROJECT_ROOT / "data/interim/fed_communications.parquet",
    "anes": PROJECT_ROOT / "data/interim/anes_2008.parquet",
    "gadarian": PROJECT_ROOT / "data/interim/gadarian.parquet",
}
INTERIM_DATA_PATHS = {
    "trump": PROJECT_ROOT / "data/interim/trump_processed.parquet",
    "yelp": PROJECT_ROOT / "data/interim/yelp_processed.parquet",
    "fed": PROJECT_ROOT / "data/interim/fed_processed.parquet",
    "anes": PROJECT_ROOT / "data/interim/anes_processed.parquet",
    "gadarian": PROJECT_ROOT / "data/interim/gadarian_processed.parquet",
}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for topic modeling."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["trump", "yelp", "fed", "anes", "gadarian"],
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum number of tokens per chunk."
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Whether to include YAML frontmatter with metadata in the text columns.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Whether to deduplicate the dataset based on cleaned text.",
    )
    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset,
        input_path=str(RAW_DATA_PATHS[args.dataset]),
        output_path=str(INTERIM_DATA_PATHS[args.dataset]),
        max_tokens=args.max_tokens,
        include_metadata=args.include_metadata,
        deduplicate=args.deduplicate,
    )


if __name__ == "__main__":
    main()
