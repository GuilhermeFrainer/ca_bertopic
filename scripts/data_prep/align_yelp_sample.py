# -*- coding: utf-8 -*-
"""Align Yelp STM dataset with sampled BERTopic embeddings."""

import sys
from pathlib import Path

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.logger_config as logger_config
from src.data import sample_from_lf

# Constants
EMBEDDINGS_PATH = PROJECT_ROOT / "data/processed/yelp_embeddings.parquet"
PROCESSED_PATH = PROJECT_ROOT / "data/interim/yelp_processed.parquet"
RAW_PATH = PROJECT_ROOT / "data/interim/yelp_reviews.parquet"
OUTPUT_EMBEDDINGS = PROJECT_ROOT / "data/processed/yelp_s10000_embeddings.parquet"
OUTPUT_UNCHUNKED = PROJECT_ROOT / "data/interim/yelp_s10000_unchunked.parquet"

SAMPLE_SIZE = 10000
RANDOM_STATE = 36201624


def remove_urls(text_expr: pl.Expr) -> pl.Expr:
    """Removes URLs and common URL residues from a Polars expression."""
    return text_expr.str.replace_all(r"https?://\S+|www\.\S+|httpstco\S+|t\.co/\S+", "")


def remove_numbers(text_expr: pl.Expr) -> pl.Expr:
    """Removes digits from a Polars expression."""
    return text_expr.str.replace_all(r"\d+", "")


def main():
    logger = logger_config.setup_logging("align_yelp_sample", PROJECT_ROOT / "logs")
    logger.info("Starting Yelp sample alignment...")

    # 1. Sample Chunks from Embeddings
    logger.info(f"Sampling {SAMPLE_SIZE} chunks from {EMBEDDINGS_PATH}...")
    emb_lf = pl.scan_parquet(EMBEDDINGS_PATH)

    # Replicate filtering from src/data.py
    emb_lf = emb_lf.filter(pl.col("clean_text").str.strip_chars() != "")

    sampled_emb_df = sample_from_lf(emb_lf, n=SAMPLE_SIZE, seed=RANDOM_STATE).collect()

    logger.info(f"Saving sampled embeddings to {OUTPUT_EMBEDDINGS}...")
    sampled_emb_df.write_parquet(OUTPUT_EMBEDDINGS)

    # 2. Map sampled chunks to original IDs
    logger.info("Mapping chunks to original document IDs...")
    # Join with yelp_processed.parquet to get the 'id' column
    # yelp_processed.parquet is 10GB, so we use scan
    proc_lf = pl.scan_parquet(PROCESSED_PATH).select(["index", "id"])

    sampled_indices = sampled_emb_df.select("index")
    mapped_ids_df = sampled_indices.join(proc_lf.collect(), on="index", how="inner")

    sampled_ids = mapped_ids_df["id"].unique().to_list()
    logger.info(f"Found {len(sampled_ids)} unique original document IDs.")

    # 3. Reconstruct Un-chunked Sample
    logger.info(f"Reconstructing un-chunked sample from {RAW_PATH}...")
    raw_lf = pl.scan_parquet(RAW_PATH)

    # Replicate logic from src/processing.py:process_dataset
    # Note: we drop 'index' and 'id' if they exist in raw to avoid confusion
    raw_lf = raw_lf.drop(["index", "id"], strict=False)

    # Standardize column names
    raw_lf = raw_lf.rename(
        {col: col.lower().replace(" ", "_") for col in raw_lf.collect_schema().names()}
    )

    # Text cleaning
    raw_lf = raw_lf.with_columns(clean_text=remove_numbers(remove_urls(pl.col("text"))))

    # Filter empty clean_text
    raw_lf = raw_lf.filter(pl.col("clean_text").str.strip_chars() != "")

    # Deduplicate (Sorting by date and keeping first)
    if "date" in raw_lf.collect_schema().names():
        raw_lf = raw_lf.sort("date").unique(
            subset=["clean_text"], keep="first", maintain_order=True
        )
    else:
        raw_lf = raw_lf.unique(subset=["clean_text"], keep="first")

    # Assign ID (this must match the ID assigned in preprocess_datasets.py)
    # The ID there is a positional index assigned after filtering and deduplication.
    clean_raw_df = raw_lf.collect().with_row_index(name="id")

    # Filter by sampled IDs
    logger.info(
        f"Filtering {len(clean_raw_df)} cleaned documents by {len(sampled_ids)} sampled IDs..."
    )
    unchunked_sample_df = clean_raw_df.filter(pl.col("id").is_in(sampled_ids))

    logger.info(f"Final un-chunked sample size: {len(unchunked_sample_df)} documents.")

    logger.info(f"Saving un-chunked sample to {OUTPUT_UNCHUNKED}...")
    unchunked_sample_df.write_parquet(OUTPUT_UNCHUNKED)

    logger.info("Alignment complete.")


if __name__ == "__main__":
    main()
