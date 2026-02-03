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
    existing_files = glob.glob(os.path.join(output_dir, "batch_*.parquet"))
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
        print("All done! Nothing to process.")
        return
        
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

        save_path = os.path.join(output_dir, output_filepath)
        chunk_with_embeddings.write_parquet(save_path)


if __name__ == "__main__":
    main()

