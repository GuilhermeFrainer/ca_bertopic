import argparse
import sys
from pathlib import Path

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.make_table import (
    generate_demsar_delta_latex_table,
    generate_demsar_delta_markdown_table,
)
from src.results_analysis import compute_demsar_delta_table

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = PROJECT_ROOT / "tables"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Demšar (2006) compliant Model-by-Metric Delta Tables with "
            "exact Wilcoxon tests and Holm-Bonferroni FWER control."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fed",
        help="Dataset name to analyze (e.g., fed, yelp). Default: fed.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="stemmed",
        choices=["stemmed", "no_stopword_removal", "keep_rep_stopwords"],
        help=(
            "Alternative preprocessing condition to compare against "
            "Default/Standard. Default: stemmed."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help=(
            "Significance level threshold (e.g. 0.10 for two-tailed or 0.05 "
            "for one-tailed). Default: 0.10."
        ),
    )
    parser.add_argument(
        "--tail",
        type=str,
        default="two-sided",
        choices=["two-sided", "greater", "less"],
        help=(
            "Wilcoxon test tail direction ('two-sided', 'greater', 'less'). "
            "Default: two-sided."
        ),
    )
    parser.add_argument(
        "--correction",
        type=str,
        default="per_metric",
        choices=["per_metric", "table", "none"],
        help=(
            "FWER correction scope ('per_metric', 'table', 'none'). "
            "Default: per_metric."
        ),
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
        help="Treat models with and without '_info0' as the same model type.",
    )
    parser.add_argument(
        "--latex",
        type=str,
        nargs="?",
        const="__STDOUT__",
        metavar="FILE",
        help=(
            "Output the results as a LaTeX table. If FILE is omitted, prints to stdout."
        ),
    )
    parser.add_argument(
        "--markdown",
        type=str,
        nargs="?",
        const="__STDOUT__",
        metavar="FILE",
        help=(
            "Output the results as a Markdown table. If FILE is omitted, "
            "prints to stdout."
        ),
    )
    parser.add_argument(
        "--save-tables",
        action="store_true",
        help="Automatically save Markdown and LaTeX tables into the tables/ directory.",
    )
    return parser.parse_args()


def load_dataset_results(dataset: str, condition: str):
    """Loads default and alternative result DataFrames for the given dataset."""
    default_files = [
        f
        for f in RESULTS_DIR.glob("*.csv")
        if f.name.lower().startswith(f"{dataset.lower()}_standard")
        or (f.name.lower().startswith(dataset.lower()) and "standard" in f.name.lower())
    ]
    if not default_files:
        # Fallback to any file starting with dataset that is not stemmed or no_stopword
        default_files = [
            f
            for f in RESULTS_DIR.glob("*.csv")
            if f.name.lower().startswith(dataset.lower())
            and "stemmed" not in f.name.lower()
            and "no_stopword" not in f.name.lower()
            and "keep_rep" not in f.name.lower()
        ]

    alt_pattern = "stemmed" if condition == "stemmed" else "no_stopword"
    alt_files = [
        f
        for f in RESULTS_DIR.glob("*.csv")
        if f.name.lower().startswith(dataset.lower()) and alt_pattern in f.name.lower()
    ]

    if not default_files:
        raise FileNotFoundError(
            f"No Default/Standard results found for dataset '{dataset}' "
            f"in {RESULTS_DIR}"
        )
    if not alt_files:
        raise FileNotFoundError(
            f"No {condition} results found for dataset '{dataset}' in {RESULTS_DIR}"
        )

    df_default = pl.concat(
        [pl.read_csv(f, infer_schema_length=None) for f in default_files],
        how="diagonal",
    )
    df_alt = pl.concat(
        [pl.read_csv(f, infer_schema_length=None) for f in alt_files],
        how="diagonal",
    )

    return df_default, df_alt


def main():
    args = parse_args()

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

    print(
        f"Executing Demsar Evaluation for dataset: '{args.dataset}', "
        f"condition: '{args.condition}'..."
    )
    print(
        f"Parameters: alpha={args.alpha}, tail={args.tail}, "
        f"correction={args.correction}"
    )

    try:
        df_default, df_alt = load_dataset_results(args.dataset, args.condition)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    results = compute_demsar_delta_table(
        df_default=df_default,
        df_alternative=df_alt,
        dataset=args.dataset,
        alpha=args.alpha,
        alternative=args.tail,
        correction=args.correction,
        exclude_clustering=exclude_clustering,
        exclude_dim_red=exclude_dim_red,
        merge_info0=args.merge_info0,
    )

    if results["df_summary"].is_empty():
        print("No matching paired models found to compute delta table.")
        return

    condition_title = (
        "Stemmed" if args.condition == "stemmed" else "No Stopword Removal"
    )

    # Generate Markdown Table
    md_table = generate_demsar_delta_markdown_table(
        delta_results=results,
        dataset=args.dataset,
        condition_name=condition_title,
    )

    # Generate LaTeX Table
    latex_table = generate_demsar_delta_latex_table(
        delta_results=results,
        dataset=args.dataset,
        condition_name=condition_title,
    )

    # CLI Outputs
    if args.markdown == "__STDOUT__" or (
        not args.latex and not args.markdown and not args.save_tables
    ):
        print("\n" + md_table)

    if args.markdown and args.markdown != "__STDOUT__":
        out_path = Path(args.markdown)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_table, encoding="utf-8")
        print(f"Markdown table written to {out_path}")

    if args.latex:
        if args.latex == "__STDOUT__":
            print("\n" + latex_table)
        else:
            out_path = Path(args.latex)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(latex_table, encoding="utf-8")
            print(f"LaTeX table written to {out_path}")

    if args.save_tables:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        cond_slug = args.condition.lower().replace(" ", "_")
        md_file = TABLES_DIR / f"demsar_delta_{args.dataset}_{cond_slug}.md"
        tex_file = TABLES_DIR / f"demsar_delta_{args.dataset}_{cond_slug}.tex"
        csv_file = TABLES_DIR / f"demsar_delta_{args.dataset}_{cond_slug}_details.csv"

        md_file.write_text(md_table, encoding="utf-8")
        tex_file.write_text(latex_table, encoding="utf-8")
        results["df_details"].write_csv(csv_file)
        print("\nTables successfully saved:")
        print(f" - Markdown: {md_file}")
        print(f" - LaTeX:    {tex_file}")
        print(f" - Details:  {csv_file}")


if __name__ == "__main__":
    main()
