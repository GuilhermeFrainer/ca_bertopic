import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pathlib import Path
import argparse
import glob
import os

DEFAULT_BATCH_SIZE = 1000
DEFAULT_MODEL = "all-MiniLM-L6-v2"

DATA_DIR = Path("data/processed")
DATASETS = {
    "yelp": DATA_DIR / "yelp_reviews.parquet",
    "trump": DATA_DIR / "trump_parsed.parquet"
}


def main():
    parser = argparse.ArgumentParser(description="Generate text embeddings for specific datasets.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=DATASETS.keys(), 
        help="The key of the dataset to process (e.g., 'yelp')."
    )
    args = parser.parse_args()
    
    try:
        process_dataset(args.dataset)
    except KeyError:
        print(f"Error: Dataset '{args.dataset}' not found in registry.")


def get_start_index(output_dir: Path, batch_size: int) -> int:
    """
    Calculates the starting row index based on existing batch files on disk.
    """
    existing_files = glob.glob(os.path.join(output_dir, "*.parquet"))
    return len(existing_files) * batch_size


def process_dataset(dataset_key: str):
    """
    Orchestrates the data loading, processing loop, and file saving.
    """
    file_path = DATASETS[dataset_key]
    
    # Setup
    output_dir = DATA_DIR / (dataset_key + "_embeddings_batches")
    os.makedirs(output_dir, exist_ok=True)
    embedding_model = SentenceTransformer(DEFAULT_MODEL)
    
    # Load Data
    df = pl.read_parquet(file_path)
    total_rows = len(df)
    
    # Check Resume Status
    start_index = get_start_index(output_dir, DEFAULT_BATCH_SIZE)
    
    print(f"Processing dataset: {dataset_key}")
    print(f"Total rows: {total_rows}")
    
    if start_index >= total_rows:
        print("All rows processed. Skipping to stitching phase...")
    else:
        print(f"Resuming from row {start_index}...")

        # Main Loop
        for i in tqdm(range(start_index, total_rows, DEFAULT_BATCH_SIZE), desc="Generating Embeddings"):
            chunk = df.slice(i, DEFAULT_BATCH_SIZE)
            texts = chunk["text"].to_list()
            output_filepath = f"{dataset_key}_batch_{i}.parquet"

            embeddings = embedding_model.encode(texts)
            
            chunk_with_embeddings = chunk.with_columns(
                pl.Series(name="embedding", values=embeddings)
            )

            save_path = output_dir / output_filepath
            chunk_with_embeddings.write_parquet(save_path)

    # Stitch files at the end
    stitch_batches(output_dir, dataset_key)


def stitch_batches(output_dir: Path, dataset_key: str):
    """
    Combines all batch files into a single parquet file.
    """
    final_output_path = DATA_DIR / f"{dataset_key}_embeddings.parquet"
    print(f"Stitching batches into {final_output_path}...")

    batch_files = list(output_dir.glob("*.parquet"))
    
    if not batch_files:
        print("No batch files found to stitch.")
        return

    # Sort files by the batch index (integer at the end of filename)
    # Filename format: "{dataset_key}_batch_{i}.parquet"
    # We split by '_' and take the last part, removing '.parquet'
    try:
        batch_files.sort(key=lambda p: int(p.stem.split('_')[-1]))
    except ValueError:
        print("Warning: Could not sort files by index. Stitching in default order.")

    # 3. Use scan_parquet + sink_parquet for memory-efficient merging
    try:
        pl.scan_parquet(batch_files).sink_parquet(final_output_path)
        print("Stitching complete.")
    except Exception as e:
        print(f"Error during stitching: {e}")


if __name__ == "__main__":
    main()

