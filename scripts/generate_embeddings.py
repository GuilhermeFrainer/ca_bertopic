import argparse
import os
import re
from pathlib import Path

import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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

    process_dataset(
        args.dataset, args.columns, args.model_name, args.batch_size
    )


def get_next_batch_index(batch_dir: Path) -> int:
    """
    Determines the next batch index by parsing filenames in the batch directory.
    Filenames are expected to be in the format 'batch_{index}.parquet'.
    """
    if not batch_dir.exists():
        return 0

    max_index = -1
    for f in batch_dir.glob("*.parquet"):
        match = re.search(r"batch_(\d+)\.parquet$", f.name)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
    return max_index + 1


def sort_batch_files(batch_files: list[Path]) -> list[Path]:
    """
    Sorts a list of batch file paths numerically based on the index in the filename.
    """

    def get_batch_number(p: Path):
        match = re.search(r"batch_(\d+)\.parquet$", p.name)
        if match:
            return int(match.group(1))
        # This case should ideally not be reached with controlled filenames
        return -1

    return sorted(batch_files, key=get_batch_number)


def process_dataset(
    dataset_name: str,
    target_columns: list[str],
    model_name: str,
    batch_size: int,
):
    """
    Orchestrates the data loading, processing loop, and file saving.
    """
    # 1. Path Updates & I/O
    input_path = Path(f"data/interim/{dataset_name}_processed.parquet")
    batch_dir = Path(f"data/interim/{dataset_name}_embeddings_batches")
    final_output_path = Path(f"data/processed/{dataset_name}_embeddings.parquet")

    os.makedirs(batch_dir, exist_ok=True)

    # 2. Embedding Optimization
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    embedding_model = SentenceTransformer(model_name, device=device)

    # Load Data
    lf = pl.scan_parquet(input_path)
    assert isinstance(lf, pl.LazyFrame)

    schema = lf.collect_schema()
    for col in target_columns:
        if col not in schema.names():
            print(
                f"Error: Column '{col}' not found in dataset '{dataset_name}'. "
                f"Available columns: {schema.names()}"
            )
            return

    total_rows = lf.select(pl.len()).collect().item() # ty: ignore
    total_batches = (total_rows + batch_size - 1) // batch_size

    # 3. Robust Checkpointing
    start_batch = get_next_batch_index(batch_dir)

    print(f"Processing dataset: {dataset_name}")
    print(f"Target columns: {target_columns}")
    print(f"Total rows: {total_rows}")
    print(f"Batch size: {batch_size}")

    if start_batch >= total_batches:
        print("All batches processed. Skipping to stitching phase...")
    else:
        print(f"Resuming from batch {start_batch}...")

        # Main Loop
        for i in tqdm(
            range(start_batch, total_batches),
            initial=start_batch,
            total=total_batches,
            desc="Generating Embeddings",
        ):
            offset = i * batch_size
            chunk = lf.slice(offset, batch_size).collect()

            # Loop through every requested column and generate embeddings
            for col in target_columns:
                texts = chunk[col].to_list() # ty: ignore
                embeddings = embedding_model.encode(
                    texts, show_progress_bar=False
                )

                # Add the new column with suffix
                chunk = chunk.with_columns( # ty: ignore
                    pl.Series(name=f"{col}_embedding", values=embeddings)
                )

            output_filepath = f"batch_{i}.parquet"
            save_path = batch_dir / output_filepath
            chunk.write_parquet(save_path) # ty: ignore

    # Stitch files at the end
    stitch_batches(batch_dir, final_output_path)


def stitch_batches(batch_dir: Path, final_output_path: Path):
    """
    Combines all batch files into a single parquet file.
    """
    print(f"Stitching batches into {final_output_path}...")

    batch_files = list(batch_dir.glob("*.parquet"))

    if not batch_files:
        print("No batch files found to stitch.")
        return

    # Sort files numerically before stitching
    sorted_batch_files = sort_batch_files(batch_files)

    try:
        pl.scan_parquet(sorted_batch_files).sink_parquet(final_output_path)
        print("Stitching complete.")
    except Exception as e:
        print(f"Error during stitching: {e}")


if __name__ == "__main__":
    main()

