
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
import yaml
from pathlib import Path

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
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ARTIFACTS_TO_REMOVE = {
    "trump": ["covfefe"],
    "yelp": [], # No specific artifacts for yelp
}
RAW_DATA_PATHS = {
    "trump": "data/raw/trump_tweets.csv",
    "yelp": "data/interim/yelp_reviews.parquet",
}
INTERIM_DATA_PATHS = {
    "trump": "data/interim/trump_processed.parquet",
    "yelp": "data/interim/yelp_processed.parquet",
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


def apply_trump_schema_and_types(df: pl.DataFrame) -> pl.DataFrame:
    """Applies Trump-specific schema and type conversions.

    This function implements the data type and naming conversions discovered
    in the 'trump_tweets_exploration.ipynb' notebook. It handles the specific
    way booleans ('t'/'f'), datetimes, and categoricals are stored in the
    raw Trump dataset.

    Args:
        df: The raw Trump dataset as a Polars DataFrame.

    Returns:
        A DataFrame with corrected data types and column names.
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
    if "content" in df.columns:
        df = df.rename({"content": "text"})
    
    # Ensure all other columns are snake_case
    df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})
    return df


def remove_urls(text_expr: pl.Expr) -> pl.Expr:
    """Removes URLs from a Polars expression.

    Args:
        text_expr: A Polars expression representing a series of strings.

    Returns:
        A Polars expression with URLs removed.
    """
    return text_expr.str.replace(r"https?://\S+", "")


def remove_artifacts(text_expr: pl.Expr, artifacts: list[str]) -> pl.Expr:
    """Removes specific artifacts from a Polars expression.

    Args:
        text_expr: A Polars expression representing a series of strings.
        artifacts: A list of strings to remove.

    Returns:
        A Polars expression with artifacts removed.
    """
    for artifact in artifacts:
        text_expr = text_expr.str.replace(artifact, "")
    return text_expr


def add_log_transformation(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Adds a log-transformed column to a Polars DataFrame.

    Args:
        df: The input Polars DataFrame.
        column: The name of the column to transform.

    Returns:
        A Polars DataFrame with the new log-transformed column.
    """
    return df.with_columns(
        (pl.col(column) + 1).log().alias(f"log_{column}")
    )


def format_as_yaml(df: pl.DataFrame, columns: list[str]) -> pl.Series:
    """Formats specified columns of a DataFrame into a YAML frontmatter string.

    Args:
        df: The input Polars DataFrame.
        columns: A list of column names to include in the YAML frontmatter.

    Returns:
        A Polars Series of YAML strings.
    """
    def to_yaml_string(row_dict):
        # Ensure proper formatting for strings and numbers
        return yaml.dump(row_dict, sort_keys=False, default_flow_style=False)

    struct_series = df.select(columns).to_struct(name="metadata")
    
    # Use map_elements for element-wise transformation on a Series
    return struct_series.map_elements(lambda x: to_yaml_string(x), return_dtype=pl.Utf8)


def chunk_text_with_overlap(
    df: pl.DataFrame,
    text_column: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    overlap_sentences: int = 1,
) -> pl.DataFrame:
    """Chunks text into smaller pieces only if it exceeds a token limit.

    Args:
        df: The input Polars DataFrame.
        text_column: The name of the text column to chunk.
        tokenizer: The Hugging Face tokenizer.
        max_tokens: The maximum number of tokens per chunk.
        overlap_sentences: The number of sentences to overlap between chunks.

    Returns:
        A new DataFrame with text chunked into multiple rows if necessary.
    """
    new_rows = []
    rows_to_chunk = 0

    for row in tqdm(df.to_dicts(), desc="Chunking text"):
        original_text = row.get(text_column, "")
        if not original_text:
            new_rows.append(row)
            continue

        total_tokens = len(tokenizer.encode(original_text, add_special_tokens=False))

        # 1. If the text is within the limit, don't chunk it.
        if total_tokens <= max_tokens:
            new_row = row.copy()
            new_row["token_count"] = total_tokens
            new_rows.append(new_row)
            continue

        # 2. If the text exceeds the limit, proceed with chunking.
        rows_to_chunk += 1
        sentences = nltk.sent_tokenize(original_text)
        if not sentences:
            continue

        current_pos = 0
        while current_pos < len(sentences):
            # Build a chunk starting from current_pos
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

            # Append the created chunk
            new_row = row.copy()
            new_row[text_column] = " ".join(chunk_sentences)
            new_row["token_count"] = chunk_tokens
            new_rows.append(new_row)

            # Move current_pos for the next chunk
            if end_pos >= len(sentences):  # Reached the end
                break
            
            # Determine overlap
            num_sentences_in_chunk = len(chunk_sentences)
            step = max(1, num_sentences_in_chunk - overlap_sentences)
            current_pos += step

    if rows_to_chunk > 0:
        logging.info(f"Chunked {rows_to_chunk} rows that exceeded the token limit.")

    if not new_rows:
        return df.with_columns(pl.lit(0).alias("token_count"))
        
    return pl.DataFrame(new_rows)


def process_dataset(
    dataset_name: str,
    input_path: str,
    output_path: str,
    tokenizer_name: str = TOKENIZER_NAME,
    max_tokens: int | None = None,
) -> pl.DataFrame:
    """Main function to process a single dataset.

    Args:
        dataset_name: The name of the dataset ('trump' or 'yelp').
        input_path: Path to the raw dataset file.
        output_path: Path to save the processed parquet file.
        tokenizer_name: The name of the Hugging Face tokenizer to use.
        max_tokens: The maximum number of tokens per chunk.

    Returns:
        The processed Polars DataFrame.
    """
    logging.info(f"Starting preprocessing for dataset: {dataset_name}")

    # 1. Load data
    logging.info(f"Loading data from {input_path}")
    if input_path.endswith("csv"):
        df = pl.read_csv(input_path)
    elif input_path.endswith("parquet"):
        df = pl.read_parquet(input_path)
    else:
        raise ValueError(f"Unexpected file format: {input_path}")

    original_row_count = df.height
    
    # 2. Standardize schema and apply data type conversions
    if dataset_name == "trump":
        df = apply_trump_schema_and_types(df)
    else:
        # Generic standardization for other datasets
        df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})
        for col in BOOLEAN_COLS.get(dataset_name, []):
             if df[col].dtype != pl.Boolean:
                df = df.with_columns((pl.col(col).str.to_lowercase() == "true").alias(col))
        for col in DATETIME_COLS.get(dataset_name, []):
            df = df.with_columns(pl.col(col).str.to_datetime())
        for col in CATEGORICAL_COLS.get(dataset_name, []):
            df = df.with_columns(pl.col(col).cast(pl.Categorical))

    # 3. Add sequential ID
    df = df.with_columns(pl.arange(0, df.height).alias("id"))

    # 5. Numerical transformations
    for col in NUMERICAL_COLS.get(dataset_name, []):
        df = add_log_transformation(df, col)

    # 6. Text preprocessing
    logging.info("Preprocessing text columns...")
    df = df.with_columns(
        clean_text=remove_urls(pl.col("text"))
    )
    df = df.with_columns(
        clean_text=remove_artifacts(pl.col("clean_text"), ARTIFACTS_TO_REMOVE[dataset_name])
    )
    df = df.with_columns(
        clean_text_lower=pl.col("clean_text").str.to_lowercase(),
    )
    df = df.with_columns(
        clean_text_lower_punctless=pl.col("clean_text_lower").str.replace_all(r"[^\w\s]", "")
    )
    
    # 7. Sentence chunking
    logging.info("Chunking text into smaller segments...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if max_tokens is None:
        max_tokens = tokenizer.model_max_length

    df_chunked = chunk_text_with_overlap(
        df,
        "clean_text",
        tokenizer,
        max_tokens=max_tokens,
        overlap_sentences=2
    )
    
    new_row_count = df_chunked.height
    logging.info(f"Added {new_row_count - original_row_count} new rows as a result of chunking.")
    df = df_chunked

    # 8. YAML frontmatter injection
    logging.info("Injecting YAML frontmatter...")
    yaml_frontmatter = format_as_yaml(df, METADATA_COLS[dataset_name])
    
    df = df.with_columns(
        clean_text_with_metadata = pl.concat_str([pl.lit('---\n'), yaml_frontmatter, pl.lit('---\n'), pl.col("clean_text")]),
        clean_text_lower_with_metadata = pl.concat_str([pl.lit('---\n'), yaml_frontmatter, pl.lit('---\n'), pl.col("clean_text_lower")]),
        clean_text_lower_punctless_with_metadata = pl.concat_str([pl.lit('---\n'), yaml_frontmatter, pl.lit('---\n'), pl.col("clean_text_lower_punctless")]),
    )
    # 9. Save output
    logging.info(f"Saving processed data to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    logging.info("Preprocessing finished successfully.")
    return df


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Preprocess datasets for topic modeling.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["trump", "yelp"],
        help="The name of the dataset to process."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="The maximum number of tokens per chunk."
    )
    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset,
        input_path=RAW_DATA_PATHS[args.dataset],
        output_path=INTERIM_DATA_PATHS[args.dataset],
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
