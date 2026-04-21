"""
Script to summarize NLP datasets using Polars LazyFrames and export to LaTeX.

This script computes core statistics for the datasets used in the
CA-BERTopic project, including basic counts, preprocessing metrics,
and descriptive statistics for token counts. It is optimized for memory
efficiency using Polars LazyFrames and avoids expensive operations
like explode().
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from great_tables import GT

# Global constants for configuration
DATASETS: List[str] = [
    "data/processed/trump_embeddings.parquet",
    "data/processed/yelp_embeddings.parquet",
]

TEXT_COL: str = "text"
CLEAN_TEXT_COL: str = "clean_text"
OUTPUT_DIR: Path = Path("tables")


def compute_dataset_summary(file_path: str) -> Dict[str, Any]:
    """Computes a statistical summary for a single dataset.

    Args:
        file_path: Path to the parquet dataset file.

    Returns:
        A dictionary containing the computed metrics.
    """
    path = Path(file_path)
    if not path.exists():
        return {"Dataset": path.name, "Error": "File not found"}

    # Initialize LazyFrame and select only required columns to save memory
    lf = pl.scan_parquet(file_path).select([TEXT_COL, CLEAN_TEXT_COL])

    # Polars expressions for efficient processing
    # Use count_matches for memory-efficient counting instead of split().list.len()
    # Tokens: sequences of non-whitespace characters
    # Sentences: occurrences of . ! ?
    summary_lf = lf.select(
        [
            pl.lit(path.name).alias("Dataset"),
            pl.len().alias("Total Documents"),
            # Preprocessing Metrics: Count empty or null clean_text documents
            (
                pl.col(CLEAN_TEXT_COL).is_null().sum()
                + (pl.col(CLEAN_TEXT_COL) == "").sum()
            ).alias("Empty Clean Docs"),
            # Token/Sentence Counts per document
            # Approximation: count spaces + 1 for tokens, punct for sentences
            # Using regex for tokens: \S+ matches non-whitespace sequences
            pl.col(TEXT_COL).str.count_matches(r"\S+").alias("token_counts"),
            pl.col(TEXT_COL).str.count_matches(r"[\.\!\?]").alias("sentence_counts"),
        ]
    ).select(
        [
            pl.col("Dataset"),
            pl.col("Total Documents"),
            pl.col("Empty Clean Docs"),
            pl.col("token_counts").sum().alias("Total Tokens"),
            pl.col("sentence_counts").sum().alias("Total Sentences"),
            pl.col("token_counts").mean().alias("Avg Tokens/Doc"),
            pl.col("token_counts").median().alias("Median Tokens/Doc"),
            pl.col("token_counts").std().alias("Std Tokens/Doc"),
        ]
    )

    # Single collect call to execute the optimized physical plan
    return summary_lf.collect().to_dicts()[0]


def main() -> None:
    """Main execution entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    results: List[Dict[str, Any]] = []

    logger.info("Analyzing datasets...")
    for dataset_path in DATASETS:
        logger.info(f"  Processing {dataset_path}...")
        try:
            summary = compute_dataset_summary(dataset_path)
            if "Error" in summary:
                logger.error(f"  {summary['Error']} for {dataset_path}")
                continue
            results.append(summary)
        except Exception as e:
            logger.error(f"  Error processing {dataset_path}: {e}")

    if not results:
        logger.warning("No results to display.")
        return

    # Create a Polars DataFrame from the results for display and Great Tables
    df_results = pl.DataFrame(results)

    # Create a professional table using Great Tables
    gt_table = (
        GT(df_results)
        .tab_header(
            title="Dataset Statistical Summary",
            subtitle="Diagnostic overview of NLP datasets for CA-BERTopic experiments",
        )
        .fmt_number(
            columns=[
                "Total Documents",
                "Total Tokens",
                "Total Sentences",
                "Empty Clean Docs",
            ],
            decimals=0,
            use_seps=True,
        )
        .fmt_number(
            columns=["Avg Tokens/Doc", "Median Tokens/Doc", "Std Tokens/Doc"],
            decimals=2,
        )
        .cols_align(align="center")
    )

    # Print a markdown-like version to the console
    logger.info("\n--- Dataset Summary Table ---")
    print(df_results)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Export to LaTeX
    latex_output_path = OUTPUT_DIR / "dataset_summary.tex"
    latex_code = gt_table.as_latex()

    with open(latex_output_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    logger.info(f"\nLaTeX table saved to: {latex_output_path}")


if __name__ == "__main__":
    main()
