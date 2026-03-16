# -*- coding: utf-8 -*-
"""Preprocesses the Trump and Yelp datasets for topic modeling.

This script executes the preprocessing logic defined in src/processing.py.
"""

import argparse
from pathlib import Path

from src.processing import process_dataset

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATHS = {
    "trump": PROJECT_ROOT / "data/raw/trump_tweets.csv",
    "yelp": PROJECT_ROOT / "data/interim/yelp_reviews.parquet",
}
INTERIM_DATA_PATHS = {
    "trump": PROJECT_ROOT / "data/interim/trump_processed.parquet",
    "yelp": PROJECT_ROOT / "data/interim/yelp_processed.parquet",
}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Preprocess datasets for topic modeling.")
    parser.add_argument("--dataset", type=str, required=True, choices=["trump", "yelp"])
    parser.add_argument("--max-tokens", type=int, help="Maximum number of tokens per chunk.")
    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset,
        input_path=str(RAW_DATA_PATHS[args.dataset]),
        output_path=str(INTERIM_DATA_PATHS[args.dataset]),
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
