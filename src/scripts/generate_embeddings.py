import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import os
import glob


OUTPUT_DIR = "data/processed/embeddings_batches"
BATCH_SIZE = 1000 # Save to disk every 1000 rows
YELP_DATASET = "data/processed/yelp_reviews.parquet"
MODEL = "all-MiniLM-L6-v2"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embedding_model = SentenceTransformer(MODEL)

    df = pl.read_parquet(YELP_DATASET)
    total_rows = len(df)

    # Check how many batches are already done
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "batch_*.parquet"))
    start_index = len(existing_files) * BATCH_SIZE

    print(f"Total rows: {total_rows}")
    print(f"Found {len(existing_files)} existing batches.")
    if start_index >= total_rows:
        print("All done! Nothing to process.")
    else:
        print(f"Resuming from row {start_index}...")
    

    for i in tqdm(range(start_index, total_rows, BATCH_SIZE), desc="Generating Embeddings"):
        chunk = df.slice(i, BATCH_SIZE)
        texts = chunk["text"].to_list()

        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        chunk_with_embeddings = chunk.with_columns(
            pl.Series(name="embedding", values=embeddings)
        )

        save_path = os.path.join(OUTPUT_DIR, f"batch_{i}.parquet")
        chunk_with_embeddings.write_parquet(save_path)


if __name__ == "__main__":
    main()

