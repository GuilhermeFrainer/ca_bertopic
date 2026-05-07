import argparse
import sys
from pathlib import Path

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.make_table import generate_best_models_latex_table
from src.results_analysis import find_best_models
from src.visualization import (
    generate_cleveland_plot,
    generate_parallel_plot,
    generate_star_plot,
)

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAME = "CAST"


def main():
    parser = argparse.ArgumentParser(
        description="Find the best performing model of each type for each metric."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., fed, yelp, trump)",
    )
    parser.add_argument(
        "--exclude-clustering",
        type=str,
        nargs="+",
        help="Clustering algorithms to exclude.",
    )
    parser.add_argument(
        "--exclude-dim-red",
        type=str,
        nargs="+",
        help="Dimensionality reduction algorithms to exclude.",
    )
    parser.add_argument(
        "--latex",
        type=str,
        nargs="?",
        const="__STDOUT__",
        help="Output the results as a LaTeX table. Optionally specify a file path.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump all results instead of only the best per type.",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help=(
            "Calculate the average performance per model type "
            "instead of finding the best."
        ),
    )
    parser.add_argument(
        "--merge-info0",
        action="store_true",
        help="Merge models with and without info0 into the same model type.",
    )
    parser.add_argument(
        "--star-plot",
        type=str,
        help=(
            "Output the results as a star plot. Specify a file path (defaults to .pdf)."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=str,
        help=(
            "Output a parallel lines plot. "
            "Specify a file path (defaults to _parallel.pdf)."
        ),
    )
    parser.add_argument(
        "--cleveland",
        type=str,
        help=(
            "Output a Cleveland dot plot. "
            "Specify a file path (defaults to _cleveland.pdf)."
        ),
    )
    args = parser.parse_args()

    dataset = args.dataset
    csv_files = list(RESULTS_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}")
        return

    all_dfs = []
    for f in csv_files:
        try:
            df = pl.read_csv(f, infer_schema_length=None)
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

    results = find_best_models(
        df,
        dataset,
        exclude_clustering=args.exclude_clustering,
        exclude_dim_red=args.exclude_dim_red,
        dump=args.dump,
        average=args.average,
        merge_info0=args.merge_info0,
    )

    if not results:
        print(f"No valid metric results found for dataset: {dataset}")
        return

    if args.star_plot:
        generate_star_plot(results, args.dump, Path(args.star_plot), model_name=MODEL_NAME)

    if args.parallel:
        generate_parallel_plot(results, args.dump, Path(args.parallel), model_name=MODEL_NAME)

    if args.cleveland:
        generate_cleveland_plot(results, args.dump, Path(args.cleveland), model_name=MODEL_NAME)

    if args.latex:
        latex_table = generate_best_models_latex_table(
            results, dataset, dump=args.dump, average=args.average
        )
        if args.latex == "__STDOUT__":
            print("\n" + latex_table)
        else:
            output_path = Path(args.latex)
            output_path.write_text(latex_table, encoding="utf-8")
            print(f"LaTeX table saved to {output_path}")
    elif not (args.star_plot or args.parallel or args.cleveland):
        title = (
            "All model configurations"
            if args.dump
            else ("Average model performance" if args.average else "Best models")
        )
        print(f"\n{title} for dataset: {dataset}")
        print("=" * (len(title) + 14 + len(dataset)))

        for metric, best_per_type in results.items():
            print(f"\nMetric: {metric}")
            print("-" * (8 + len(metric)))

            # Print results in a nice format
            for row in best_per_type.iter_rows(named=True):
                model_name = row["best_model_name"]
                max_value = row["max_value"]
                print(f"  {model_name:<50} | {max_value:>8.4f}")


if __name__ == "__main__":
    main()
