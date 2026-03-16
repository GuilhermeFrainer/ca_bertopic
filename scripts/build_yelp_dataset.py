from pathlib import Path
import polars as pl
import argparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RELEVANT_COLS = [
    "date",
    "stars",
    "text",
    "review_count",
    "average_stars",
    "yelping_since",
    "state",
    "stars_business",
    "review_count_business"
]


def main():
    parser = argparse.ArgumentParser(description="Process JSON files into a single Parquet file.")

    parser.add_argument(
        "--skip-convert", 
        action="store_true", 
        help="Skip the JSON-to-Parquet conversion step and only perform the merge."
    )
    
    args = parser.parse_args()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_convert:
        write_parquet_files(data_dir=DATA_DIR)

    business_lf = pl.scan_parquet(DATA_DIR / "yelp_academic_dataset_business.parquet")
    review_lf = pl.scan_parquet(DATA_DIR / "yelp_academic_dataset_review.parquet")
    user_lf = pl.scan_parquet(DATA_DIR / "yelp_academic_dataset_user.parquet")

    tmp_lf = review_lf.join(other=user_lf, on="user_id", suffix="_user")
    full_df = tmp_lf.join(other=business_lf, on="business_id", suffix="_business")

    full_df.sink_parquet(DATA_DIR / "full_yelp_reviews.parquet")

    lf = pl.scan_parquet(DATA_DIR / "full_yelp_reviews.parquet")
    lf = change_lf(lf, select_cols=RELEVANT_COLS)
    lf = lf.with_row_index()
    lf.sink_parquet(INTERIM_DIR / "yelp_reviews.parquet")


def write_parquet_files(
    data_dir: Path,
    filename_prefix: str = "yelp_academic_dataset_",
    files: list[str] = ["business", "review", "user"]
):
    """
    Converts the original JSON files into parquet files for faster reading and processing.
    Datetime columns are parsed as such.
    No other changes are made to the files.
    """
    for file in files:
        filename = filename_prefix + file
        df = pl.read_ndjson(data_dir / (filename + ".json"))

        if file == "review":
            df = df.with_columns(
                pl.col("date").str.to_datetime()
            )
        elif file == "user":
            df = df.with_columns(
                pl.col("yelping_since").str.to_datetime()
            )

        df.write_parquet(data_dir / (filename + ".parquet"))
        del df


def change_lf(lf: pl.LazyFrame, select_cols: list[str]) -> pl.LazyFrame:
    """
    Makes changes to the LazyFrame such as selecting only relevant columns
    and renaming others.
    """
    return lf.select(select_cols).rename({
        "review_count_business": "business_review_count",
        "stars_business": "business_stars",
        "average_stars": "user_average_stars",
        "review_count": "user_review_count"
    })


if __name__ == "__main__":
    main()

