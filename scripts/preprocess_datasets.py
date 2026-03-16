# -*- coding: utf-8 -*-
"""Preprocesses the Trump and Yelp datasets for topic modeling.

This script provides a set of functions to clean, transform, and standardize
the datasets. It follows a functional programming paradigm to ensure that
the operations are deterministic and modular.

Example:
    python src/preprocess_datasets.py --dataset trump
    python src/preprocess_datasets.py --dataset yelp
"""

import argparse
import logging
import tempfile
import yaml
from pathlib import Path
from typing import Union

import polars as pl
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 10000  # Process 10,000 rows at a time
ARTIFACTS_TO_REMOVE = {
    "trump": ["covfefe"],
    "yelp": [], # No specific artifacts for yelp
}
RAW_DATA_PATHS = {
    "trump": PROJECT_ROOT / "data/raw/trump_tweets.csv",
    "yelp": PROJECT_ROOT / "data/interim/yelp_reviews.parquet",
}
INTERIM_DATA_PATHS = {
    "trump": PROJECT_ROOT / "data/interim/trump_processed.parquet",
    "yelp": PROJECT_ROOT / "data/interim/yelp_processed.parquet",
}
NUMERICAL_COLS = {
    "trump": ["retweets", "favorites"],
    "yelp": ["user_review_count", "business_review_count"],
}
CATEGORICAL_COLS = {
    "trump": ["device"],
    "yelp": ["state"],
}
BOOLEAN_COLS = {
    "trump": ["is_retweet", "is_deleted", "is_flagged"],
    "yelp": [],
}
DATETIME_COLS = {
    "trump": ["date"],
    "yelp": [],
}
METADATA_COLS = {
    "trump": ["device", "log_retweets", "log_favorites"],
    "yelp": ["state", "log_user_review_count", "log_business_review_count"],
}


Frame = Union[pl.DataFrame, pl.LazyFrame]


def apply_trump_schema_and_types(df: Frame) -> Frame:
    """Applies Trump-specific schema and type conversions.

    This function implements the data type and naming conversions discovered
    in the 'trump_tweets_exploration.ipynb' notebook. It handles the specific
    way booleans ('t'/'f'), datetimes, and categoricals are stored in the
    raw Trump dataset.

    Args:
        df: The raw Trump dataset as a Polars DataFrame or LazyFrame.

    Returns:
        A Frame with corrected data types and column names.
    """
    df = df.with_columns(
        # Convert boolean-like columns ('t'/'f') to actual booleans
        (pl.col("isRetweet").str.to_lowercase() == "t").alias("is_retweet"),
        (pl.col("isDeleted").str.to_lowercase() == "t").alias("is_deleted"),
        (pl.col("isFlagged").str.to_lowercase() == "t").alias("is_flagged"),

        # Convert date column using the specific format
        pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S"),

        # Cast device to categorical
        pl.col("device").cast(pl.Categorical),
    ).drop("isRetweet", "isDeleted", "isFlagged")

    # Rename 'content' to 'text' for consistency across datasets
    # In lazy mode, we check against the schema
    schema = df.schema
    if "content" in schema:
        df = df.rename({"content": "text"})
    
    # Ensure all other columns are snake_case
    df = df.rename({col: col.lower().replace(" ", "_") for col in schema})
    return df


def remove_urls(text_expr: pl.Expr) -> pl.Expr:
    """Removes URLs from a Polars expression."""
    return text_expr.str.replace_all(r"https?://\S+", "")


def remove_artifacts(text_expr: pl.Expr, artifacts: list[str]) -> pl.Expr:
    """Removes specific artifacts from a Polars expression."""
    if not artifacts:
        return text_expr
    return text_expr.str.replace_all("|".join(artifacts), "")


def add_log_transformation(df: Frame, column: str) -> Frame:
    """Adds a log-transformed column to a Polars DataFrame or LazyFrame."""
    return df.with_columns(
        (pl.col(column) + 1).log().alias(f"log_{column}")
    )


def format_as_yaml(df: pl.DataFrame, columns: list[str]) -> pl.Series:
    """Formats specified columns of a DataFrame into a YAML frontmatter string."""
    def to_yaml_string(row_dict):
        return yaml.dump(row_dict, sort_keys=False, default_flow_style=False)

    struct_series = df.select(columns).to_struct(name="metadata")
    return struct_series.map_elements(lambda x: to_yaml_string(x), return_dtype=pl.Utf8)


def chunk_text_with_overlap(
    df: pl.DataFrame,
    text_column: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    overlap_sentences: int = 1,
) -> pl.DataFrame:
    """Chunks text into smaller pieces only if it exceeds a token limit."""
    new_rows = []
    rows_to_chunk = 0

    # Use to_dicts() for efficient row iteration
    for row in df.to_dicts():
        original_text = row.get(text_column, "")
        if not original_text:
            new_rows.append(row)
            continue

        total_tokens = len(tokenizer.encode(original_text, add_special_tokens=False))

        if total_tokens <= max_tokens:
            new_row = row.copy()
            new_row["token_count"] = total_tokens
            new_rows.append(new_row)
            continue

        rows_to_chunk += 1
        sentences = nltk.sent_tokenize(original_text)
        if not sentences:
            continue

        current_pos = 0
        while current_pos < len(sentences):
            chunk_sentences = []
            chunk_tokens = 0
            end_pos = current_pos
            
            while end_pos < len(sentences):
                sent = sentences[end_pos]
                sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                
                if chunk_tokens + sent_tokens > max_tokens and chunk_sentences:
                    break
                
                chunk_sentences.append(sent)
                chunk_tokens += sent_tokens
                end_pos += 1

            new_row = row.copy()
            new_row[text_column] = " ".join(chunk_sentences)
            new_row["token_count"] = chunk_tokens
            new_rows.append(new_row)

            if end_pos >= len(sentences):
                break
            
            step = max(1, len(chunk_sentences) - overlap_sentences)
            current_pos += step

    if rows_to_chunk > 0:
        logging.info(f"Chunked {rows_to_chunk} rows within the batch.")

    if not new_rows:
        return df.with_columns(pl.lit(0, dtype=pl.Int32).alias("token_count"))
        
    return pl.from_dicts(new_rows, schema={**df.schema, "token_count": pl.Int32})


def process_dataset(
    dataset_name: str,
    input_path: str,
    output_path: str,
    tokenizer_name: str = TOKENIZER_NAME,
    max_tokens: int | None = None,
) -> None:
    """Main function to process a single dataset using lazy evaluation and batching."""
    logging.info(f"Starting preprocessing for dataset: {dataset_name}")

    logging.info(f"Scanning data from {input_path}")
    if input_path.endswith("csv"):
        lf = pl.scan_csv(input_path)
    else:
        lf = pl.scan_parquet(input_path)

    if dataset_name == "trump":
        lf = apply_trump_schema_and_types(lf)
    else:
        lf = lf.rename({col: col.lower().replace(" ", "_") for col in lf.schema})

    for col in NUMERICAL_COLS.get(dataset_name, []):
        lf = add_log_transformation(lf, col)

    logging.info("Applying lazy text preprocessing...")
    lf = lf.with_columns(
        clean_text=remove_urls(pl.col("text"))
    ).with_columns(
        clean_text=remove_artifacts(pl.col("clean_text"), ARTIFACTS_TO_REMOVE.get(dataset_name, []))
    ).with_columns(
        clean_text_lower=pl.col("clean_text").str.to_lowercase(),
    ).with_columns(
        clean_text_lower_punctless=pl.col("clean_text_lower").str.replace_all(r"[^\w\s]", "")
    )
    
    logging.info("Starting batch processing for chunking and YAML injection...")
    
    total_rows = lf.select(pl.len()).collect().item()
    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if max_tokens is None:
        max_tokens = tokenizer.model_max_length

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir)
        
        process_bar = tqdm(range(num_batches), desc="Preprocessing and Chunking Batches")
        for i in process_bar:
            batch_df = lf.slice(i * BATCH_SIZE, BATCH_SIZE).collect()
            
            original_rows = batch_df.height
            batch_df = batch_df.with_columns(
                pl.arange(i * BATCH_SIZE, i * BATCH_SIZE + original_rows).alias("id")
            )

            chunked_df = chunk_text_with_overlap(
                batch_df, "clean_text", tokenizer, max_tokens=max_tokens, overlap_sentences=2
            )
            
            process_bar.set_postfix_str(f"Original: {original_rows}, Chunked: {chunked_df.height}")

            yaml_frontmatter = format_as_yaml(chunked_df, METADATA_COLS[dataset_name])
            
            final_batch_df = chunked_df.with_columns(
                clean_text_with_metadata=pl.concat_str([pl.lit('---\n'), yaml_frontmatter, pl.lit('---\n'), pl.col("clean_text")]),
                clean_text_lower_with_metadata=pl.concat_str([pl.lit('---\n'), yaml_frontmatter, pl.lit('---\n'), pl.col("clean_text_lower")]),
                clean_text_lower_punctless_with_metadata=pl.concat_str([pl.lit('---\n'), yaml_frontmatter, pl.lit('---\n'), pl.col("clean_text_lower_punctless")]),
            )
            final_batch_df.write_parquet(batch_dir / f"batch_{i}.parquet")

        logging.info(f"Stitching batches and saving to {output_path}")
        batch_files = sorted(batch_dir.glob("*.parquet"), key=lambda p: int(p.stem.split('_')[-1]))
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pl.scan_parquet(batch_files).sink_parquet(output_path)

    logging.info("Preprocessing finished successfully.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Preprocess datasets for topic modeling.")
    parser.add_argument("--dataset", type=str, required=True, choices=["trump", "yelp"])
    parser.add_argument("--max-tokens", type=int, help="Maximum number of tokens per chunk.")
    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset,
        input_path=RAW_DATA_PATHS[args.dataset],
        output_path=INTERIM_DATA_PATHS[args.dataset],
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
