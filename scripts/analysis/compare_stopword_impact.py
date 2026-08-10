import argparse
import sys
from pathlib import Path

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.make_table import generate_stopword_impact_latex_table
from src.results_analysis import compute_stopword_impact

RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    parser = argparse.ArgumentParser(
        description="Compute and showcase metric changes between runs with and without stopword removal."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fed",
        help="Dataset name to analyze (default: fed).",
    )
    parser.add_argument(
        "--exclude-clustering",
        type=str,
        nargs="+",
        default=["kmeans", "spherical_kmeans"],
        help="Clustering algorithms to exclude. Pass 'none' to disable.",
    )
    parser.add_argument(
        "--exclude-dim-red",
        type=str,
        nargs="+",
        default=["pca"],
        help="Dimensionality reduction algorithms to exclude. Pass 'none' to disable.",
    )
    parser.add_argument(
        "--merge-info0",
        action="store_true",
        help="Treat models with and without the 'info0' parameter as the same model type.",
    )
    parser.add_argument(
        "--latex",
        type=str,
        nargs="?",
        const="__STDOUT__",
        metavar="FILE",
        help="Output the results as a LaTeX table. If FILE is omitted, prints to stdout.",
    )

    args = parser.parse_args()

    exclude_clustering = (
        None
        if "none" in [c.lower() for c in args.exclude_clustering]
        else args.exclude_clustering
    )
    exclude_dim_red = (
        None
        if "none" in [d.lower() for d in args.exclude_dim_red]
        else args.exclude_dim_red
    )

    # 1. Load all CSV result files for the dataset
    all_files = [
        f
        for f in RESULTS_DIR.glob("*.csv")
        if f.name.lower().startswith(args.dataset.lower())
    ]
    if not all_files:
        print(
            f"No result CSV files found for dataset '{args.dataset}' in {RESULTS_DIR}"
        )
        return

    remove_dfs = []
    keep_dfs = []

    for f in all_files:
        fname = f.name.lower()
        if "stemmed" in fname:
            continue

        try:
            df = pl.read_csv(f, infer_schema_length=None)
            if df.is_empty():
                continue

            # Categorize based on stopword_removal column if present, or filename pattern
            if "stopword_removal" in df.columns:
                status = df["stopword_removal"][0]
                if status in ("remove_rep_stopwords", "default"):
                    remove_dfs.append(df)
                elif status in ("keep_rep_stopwords", "none", "no_stopword_removal"):
                    keep_dfs.append(df)
            else:
                if "keep_rep_stopwords" in fname or "no_stopword" in fname:
                    keep_dfs.append(df)
                else:
                    remove_dfs.append(df)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")

    if not remove_dfs:
        print(
            f"No 'remove_rep_stopwords' DataFrames loaded for dataset '{args.dataset}'."
        )
        return
    if not keep_dfs:
        print(
            f"No 'keep_rep_stopwords' DataFrames loaded for dataset '{args.dataset}'."
        )
        return

    df_remove = pl.concat(remove_dfs, how="diagonal")
    df_keep = pl.concat(keep_dfs, how="diagonal")

    # 3. Compute impact deltas (remove_rep_stopwords - keep_rep_stopwords)
    results = compute_stopword_impact(
        df_remove,
        df_keep,
        dataset=args.dataset,
        exclude_clustering=exclude_clustering,
        exclude_dim_red=exclude_dim_red,
        merge_info0=args.merge_info0,
    )

    if not results:
        print(f"No matching run pairs found for dataset '{args.dataset}'.")
        return

    # 4. Output results
    if args.latex:
        latex_table = generate_stopword_impact_latex_table(
            results, dataset=args.dataset
        )
        if args.latex == "__STDOUT__":
            print("\n" + latex_table)
        else:
            out_path = Path(args.latex)
            out_path.write_text(latex_table, encoding="utf-8")
            print(f"LaTeX table saved to {out_path}")
    else:
        print(f"\nStopword Removal Metric Impact for dataset: {args.dataset}")
        print("=" * 60)
        for metric, df_m in results.items():
            print(f"\nMetric: {metric}")
            print("-" * (8 + len(metric)))
            for row in df_m.iter_rows(named=True):
                mt = row["model_type"]
                mean_d = row["mean_delta"]
                std_d = row.get("std_delta", 0.0)
                n = row.get("n_pairs", 0)
                sign = "+" if mean_d > 0 else ""
                print(f"  {mt:<35} | {sign}{mean_d:>7.4f} ± {std_d:<6.4f} (n={n})")


if __name__ == "__main__":
    main()
