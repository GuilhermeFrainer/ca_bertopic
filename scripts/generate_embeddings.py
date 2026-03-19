# -*- coding: utf-8 -*-
"""Generate sentence embeddings for preprocessed datasets."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import process_dataset


def main():
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(
        description="Generate sentence embeddings for preprocessed datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to process (e.g., 'yelp', 'trump').",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["text"],
        help="List of columns to embed (e.g., --columns text title). Default: 'text'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="The SentenceTransformer model to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="The number of rows to process in each batch.",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    input_path = PROJECT_ROOT / f"data/interim/{args.dataset}_processed.parquet"
    batch_dir = PROJECT_ROOT / f"data/interim/{args.dataset}_embeddings_batches"
    final_output_path = PROJECT_ROOT / f"data/processed/{args.dataset}_embeddings.parquet"

    process_dataset(
        input_path=input_path,
        batch_dir=batch_dir,
        final_output_path=final_output_path,
        target_columns=args.columns,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
