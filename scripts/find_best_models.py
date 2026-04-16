import polars as pl
import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results_analysis import find_best_models

RESULTS_DIR = PROJECT_ROOT / "results"

def main():
    parser = argparse.ArgumentParser(description="Find the best performing model of each type for each metric.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., fed, yelp, trump)")
    args = parser.parse_args()

    dataset = args.dataset
    csv_files = list(RESULTS_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}")
        return

    all_dfs = []
    for f in csv_files:
        try:
            df = pl.read_csv(f)
            # Filter by dataset early if possible
            if "dataset_name" in df.columns:
                df = df.filter(pl.col("dataset_name") == dataset)
                if not df.is_empty():
                    all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        print(f"No results found for dataset: {dataset}")
        return

    # Combine all dataframes
    df = pl.concat(all_dfs, how="diagonal")

    results = find_best_models(df, dataset)

    if not results:
        print(f"No valid metric results found for dataset: {dataset}")
        return

    print(f"\nBest models for dataset: {dataset}")
    print("=" * (24 + len(dataset)))

    for metric, best_per_type in results.items():
        print(f"\nMetric: {metric}")
        print("-" * (8 + len(metric)))
        
        # Print results in a nice format
        for row in best_per_type.iter_rows(named=True):
            model_type = row["model_type"]
            max_value = row["max_value"]
            best_model_name = row["best_model_name"]
            print(f"  {model_type:<30} | {max_value:>8.4f} | ({best_model_name})")

if __name__ == "__main__":
    main()
