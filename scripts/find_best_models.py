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

MODEL_SHORT_NAMES = {
    "mv_co_reg_spectral": "{model_name}_1",
    "mv_co_reg_spectral_info0": "{model_name}_1-info0",
    "mv_spectral": "{model_name}_2",
    "mv_spectral_info0": "{model_name}_2-info0",
    "aligned_umap": "{model_name}_3",
    "baseline": "BERTopic_1",
    "umap_spectral": "BERTopic_2",
    "append_umap": "Naive",
}

LABEL_DESCRIPTIONS = {
    "{model_name}_1": "{model_name} with Co-Reg. Spectral Clustering",
    "{model_name}_1-info0": "Same as above with the info0 parameter",
    "{model_name}_2": "{model_name} with MV Spectral Clustering",
    "{model_name}_2-info0": "Same as above with the info0 parameter",
    "{model_name}_3": "{model_name} with Aligned UMAP",
    "BERTopic_1": "Default BERTopic",
    "BERTopic_2": "BERTopic with Spectral",
    "Naive": "Naive baseline",
}


def generate_label_table():
    """Generates a LaTeX table for model label descriptions."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Model Label Descriptions}",
        "\\label{tab:label_descriptions}",
        "\\begin{tabular}{ll}",
        "    \\toprule",
        "    Label & Description \\\\",
        "    \\midrule",
    ]
    for label, desc in LABEL_DESCRIPTIONS.items():
        # Use math mode for labels to support subscripts
        fmt_label = label.format(model_name=MODEL_NAME)
        if "_" in fmt_label:
            base, sub = fmt_label.split("_", 1)
            fmt_label = f"$\\text{{{base}}}_{{{sub}}}$"
        
        fmt_desc = desc.format(model_name=MODEL_NAME)
        lines.append(f"    {fmt_label} & {fmt_desc} \\\\")

    lines.extend([
        "    \\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Find the best performing model of each type for each metric."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
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
    parser.add_argument(
        "--label-table",
        action="store_true",
        help="Output the label description table as a LaTeX table.",
    )
    parser.add_argument(
        "--suppress-nulls",
        action="store_true",
        help="Filter out rows that contain NaNs for any of the metric columns.",
    )
    args = parser.parse_args()

    if args.label_table:
        print(generate_label_table())
        return

    if not args.dataset:
        parser.error("--dataset is required unless --label-table is provided.")

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
        suppress_nulls=args.suppress_nulls,
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
