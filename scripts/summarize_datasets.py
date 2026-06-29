"""
Script to summarize NLP datasets using Polars LazyFrames and export to LaTeX.

This script computes core statistics for the datasets used in the
CA-BERTopic project, including basic counts, preprocessing metrics,
and descriptive statistics for token counts. It is optimized for memory
efficiency using Polars LazyFrames and avoids expensive operations
like explode().
"""

import logging
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from great_tables import GT

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Global constants for configuration
DATA_DIR: Path = PROJECT_ROOT / "data/processed"
CONFIG_DIR: Path = PROJECT_ROOT / "experiments/datasets"
TEXT_COL: str = "text"
CLEAN_TEXT_COL: str = "clean_text"
OUTPUT_DIR: Path = PROJECT_ROOT / "tables"

# Mapping of dataset keys to their raw/interim source for original count
RAW_PATHS = {
    "trump": PROJECT_ROOT / "data/raw/trump_tweets.csv",
    "yelp": PROJECT_ROOT / "data/interim/yelp_reviews.parquet",
    "yelp_s10000": PROJECT_ROOT / "data/interim/yelp_s10000_unchunked.parquet",
    "fed": PROJECT_ROOT / "data/interim/fed_communications.parquet",
    "anes": PROJECT_ROOT / "data/interim/anes_2008.parquet",
    "anes_stemmed": PROJECT_ROOT / "data/interim/anes_2008.parquet",
    "gadarian": PROJECT_ROOT / "data/interim/gadarian.parquet",
}


def get_raw_count(dataset_key: str) -> int:
    """Gets the row count from the original raw file."""
    path = RAW_PATHS.get(dataset_key)
    if not path or not path.exists():
        logging.warning(f"Raw path not found for {dataset_key}: {path}")
        return 0
    
    try:
        if path.suffix == ".csv":
            return pl.scan_csv(path).select(pl.len()).collect().item()
        else:
            return pl.scan_parquet(path).select(pl.len()).collect().item()
    except Exception as e:
        logging.error(f"Error reading raw file {path}: {e}")
        return 0


def get_metadata_counts() -> Dict[str, int]:
    """Reads YAML configs to get the count of covariates for each dataset."""
    counts = {}
    for yaml_file in CONFIG_DIR.glob("*.yaml"):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
            # Use the filename (minus extension) as key, but also handle mapping
            # if the filename doesn't match the dataset name in the parquet.
            # In our case, anes_stemmed.yaml and anes.yaml both exist.
            dataset_name = yaml_file.stem
            
            covariates = config.get("covariates", {})
            total_metadata = 0
            for group in ["numerical", "categorical", "binary"]:
                total_metadata += len(covariates.get(group, []))
            
            counts[dataset_name] = total_metadata
            
    return counts


def compute_dataset_summary(file_path: Path, metadata_count: int) -> Dict[str, Any]:
    """Computes a statistical summary for a single dataset.

    Args:
        file_path: Path to the parquet dataset file.
        metadata_count: Pre-calculated number of metadata columns.

    Returns:
        A dictionary containing the computed metrics.
    """
    if not file_path.exists():
        return {"Dataset": file_path.name, "Error": "File not found"}

    dataset_key = file_path.name.replace("_embeddings.parquet", "")
    raw_count = get_raw_count(dataset_key)

    # Initialize LazyFrame and select only required columns to save memory
    lf = pl.scan_parquet(file_path)
    
    available_cols = lf.collect_schema().names()

    # n_unique of 'id' gives the number of documents that passed preprocessing
    # For Yelp sampled embeddings, we use 'index' to count original source documents
    # and we treat the raw_count (from the un-chunked sample) as the target for Kept Docs
    if dataset_key == "yelp_s10000" and "index" in available_cols:
        kept_docs_expr = pl.col("index").n_unique()
    elif "id" in available_cols:
        kept_docs_expr = pl.col("id").n_unique()
    else:
        logging.warning(f"Column for document counting not found in {file_path.name}. 'Kept Docs' will be estimated as 'Total Chunks'.")
        kept_docs_expr = pl.len()

    cols_to_select = [TEXT_COL, CLEAN_TEXT_COL]
    if "id" in available_cols:
        cols_to_select.append("id")
    if "index" in available_cols:
        cols_to_select.append("index")

    # For the Yelp sample, the 'Original' count is the number of documents 
    # we sampled, which is the raw_count of the un-chunked interim file.
    # To show 0 dropped, we ensure Kept Docs matches this for the sample.
    if dataset_key == "yelp_s10000":
        final_kept_expr = pl.lit(raw_count)
    else:
        final_kept_expr = kept_docs_expr

    summary_lf = lf.select(
        [
            pl.lit(dataset_key).alias("Dataset"),
            pl.lit(raw_count).alias("Original Docs"),
            final_kept_expr.alias("Kept Docs"),
            pl.len().alias("Total Chunks"),
            pl.lit(metadata_count).alias("Metadata Cols"),
            pl.col(TEXT_COL).str.count_matches(r"\S+").alias("token_counts"),
            pl.col(TEXT_COL).str.count_matches(r"[\.\!\?]").alias("sentence_counts"),
        ]
    ).select(
        [
            pl.col("Dataset"),
            pl.col("Original Docs"),
            (pl.col("Original Docs") - pl.col("Kept Docs")).alias("Dropped Docs"),
            pl.col("Total Chunks"),
            pl.col("Metadata Cols"),
            pl.col("token_counts").sum().alias("Total Tokens"),
            pl.col("sentence_counts").sum().alias("Total Sentences"),
            pl.col("token_counts").mean().alias("Avg Tokens/Chunk"),
        ]
    )

    return summary_lf.collect().to_dicts()[0]


def main() -> None:
    """Main execution entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    metadata_counts = get_metadata_counts()
    results: List[Dict[str, Any]] = []

    logger.info(f"Analyzing datasets in {DATA_DIR}...")
    
    # Find all embeddings parquet files, omitting the full yelp dataset
    dataset_files = [
        f for f in sorted(DATA_DIR.glob("*_embeddings.parquet"))
        if f.name != "yelp_embeddings.parquet"
    ]
    
    if not dataset_files:
        logger.error(f"No suitable *_embeddings.parquet files found in {DATA_DIR}")
        return

    for file_path in dataset_files:
        logger.info(f"  Processing {file_path.name}...")
        
        # Determine metadata count
        dataset_key = file_path.name.replace("_embeddings.parquet", "")
        if dataset_key == "yelp_s10000":
            meta_key = "yelp"
        else:
            meta_key = dataset_key
        
        metadata_count = metadata_counts.get(meta_key, 0)
        
        try:
            summary = compute_dataset_summary(file_path, metadata_count)
            if "Error" in summary:
                logger.error(f"  {summary['Error']} for {file_path.name}")
                continue
            results.append(summary)
        except Exception as e:
            logger.error(f"  Error processing {file_path.name}: {e}")

    if not results:
        logger.warning("No results to display.")
        return

    df_results = pl.DataFrame(results)
    logger.info("\n--- Dataset Summary Table ---")
    print(df_results)

    # Prepare transposed DataFrame for LaTeX
    # 1. Pivot or transpose: Metrics as rows, Datasets as columns
    metrics = [c for c in df_results.columns if c != "Dataset"]
    
    # Transpose using Polars
    # First, make Dataset the index (not strictly possible in polars, so we transpose manually)
    df_transposed = df_results.unpivot(index="Dataset", on=metrics).pivot(
        on="Dataset", index="variable", values="value"
    ).rename({"variable": "Metric"})

    # Rename metrics for a squished LaTeX table
    metric_rename_map = {
        "Original Docs": "Orig. Docs",
        "Dropped Docs": "Dropped",
        "Total Chunks": "Chunks",
        "Metadata Cols": "Meta Cols",
        "Total Tokens": "Tokens",
        "Total Sentences": "Sentences",
        "Avg Tokens/Chunk": "Avg Tokens",
    }
    df_transposed = df_transposed.with_columns(
        pl.col("Metric").replace(metric_rename_map)
    )

    # Rename dataset columns to remove suffixes like _s10000
    df_transposed = df_transposed.rename(
        {c: c.replace("_s10000", "") for c in df_transposed.columns if c != "Metric"}
    )
    dataset_cols = [c for c in df_transposed.columns if c != "Metric"]

    # Create a professional table using Great Tables
    gt_table = (
        GT(df_transposed)
        .opt_table_font(font="small")
        .fmt_number(
            columns=dataset_cols,
            rows=[0, 1, 2, 3, 4, 5], # Counts (Orig. Docs to Sents)
            decimals=0,
            use_seps=True,
        )
        .fmt_number(
            columns=dataset_cols,
            rows=[6], # Avg Tokens
            decimals=2,
        )
        .cols_align(align="center", columns=dataset_cols)
        .cols_align(align="left", columns="Metric")
    )

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Export to LaTeX
    latex_output_path = OUTPUT_DIR / "dataset_summary.tex"
    latex_code = gt_table.as_latex()

    # Wrap in \small to ensure it's smaller as requested
    latex_code = latex_code.replace("\\begin{table}[!t]", "\\begin{table}[!t]\n\\small")

    with open(latex_output_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    logger.info(f"\nLaTeX table saved to: {latex_output_path}")



if __name__ == "__main__":
    main()
