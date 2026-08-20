import argparse
import sys
from pathlib import Path

import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.make_table import (
    generate_demsar_all_vs_all_latex_table,
    generate_demsar_all_vs_all_report,
    generate_pairwise_delta_latex_matrix,
)
from src.results_analysis import compute_demsar_all_vs_all

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = PROJECT_ROOT / "tables"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Demšar (2006) All-vs-All Model Comparisons across topic model "
            "evaluation metrics with Friedman/Iman-Davenport tests, Nemenyi Critical "
            "Difference, Demšar cliques, and Holm-Bonferroni pairwise testing."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["fed"],
        help=(
            "Dataset(s) to analyze (e.g. 'fed', 'yelp', 'fed yelp', or 'all'). "
            "Default: fed."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="standard",
        choices=["standard", "stemmed", "no_stopword_removal"],
        help=(
            "Preprocessing condition of results files to load "
            "('standard', 'stemmed', 'no_stopword_removal'). Default: standard."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level threshold alpha (e.g. 0.05). Default: 0.05.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["u_mass", "c_v", "c_npmi", "irbo", "topic_diversity"],
        help=(
            "Evaluation metrics to test. "
            "Default: u_mass c_v c_npmi irbo topic_diversity."
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
            "Output the results as LaTeX tables. If FILE is omitted, prints to stdout."
        ),
    )
    parser.add_argument(
        "--markdown",
        type=str,
        nargs="?",
        const="__STDOUT__",
        metavar="FILE",
        help=(
            "Output the results as Markdown report. If FILE is omitted, "
            "prints to stdout."
        ),
    )
    parser.add_argument(
        "--save-tables",
        action="store_true",
        help="Automatically save Markdown and LaTeX tables into the tables/ directory.",
    )
    return parser.parse_args()


def load_result_files(datasets: list[str], condition: str) -> pl.DataFrame:
    """Loads matching result CSVs for the specified datasets and condition."""
    all_files = list(RESULTS_DIR.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No result CSV files found in {RESULTS_DIR}")

    selected_files = []
    is_all = "all" in [d.lower() for d in datasets]

    for f in all_files:
        fname = f.name.lower()
        if (
            condition == "standard"
            and "stemmed" not in fname
            and "no_stopword" not in fname
        ):
            if is_all or any(d.lower() in fname for d in datasets):
                selected_files.append(f)
        elif condition == "stemmed" and "stemmed" in fname:
            if is_all or any(d.lower() in fname for d in datasets):
                selected_files.append(f)
        elif condition == "no_stopword_removal" and "no_stopword" in fname:
            if is_all or any(d.lower() in fname for d in datasets):
                selected_files.append(f)

    if not selected_files:
        raise FileNotFoundError(
            f"No result files matching datasets={datasets} and "
            f"condition={condition} in {RESULTS_DIR}"
        )

    dfs = [pl.read_csv(f, infer_schema_length=None) for f in selected_files]
    return pl.concat(dfs, how="diagonal")


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

    ds_label = ", ".join(args.dataset).upper()
    print(
        f"Executing Demšar All-vs-All Model Comparison for dataset(s): {ds_label}, "
        f"condition: '{args.condition}', alpha={args.alpha}..."
    )

    try:
        df = load_result_files(args.dataset, args.condition)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    filter_ds = None if "all" in [d.lower() for d in args.dataset] else args.dataset
    if filter_ds and len(filter_ds) == 1:
        filter_ds = filter_ds[0]

    results = compute_demsar_all_vs_all(
        df=df,
        dataset=filter_ds,
        metrics=args.metrics,
        alpha=args.alpha,
        exclude_clustering=exclude_clustering,
        exclude_dim_red=exclude_dim_red,
        merge_info0=args.merge_info0,
    )

    if not results.get("metrics"):
        print("No evaluation metrics could be computed (insufficient blocks/models).")
        return

    # Generate full Markdown report
    md_report = generate_demsar_all_vs_all_report(results, dataset_label=ds_label)

    # Print or save output
    if args.markdown == "__STDOUT__" or (
        not args.latex and not args.markdown and not args.save_tables
    ):
        print("\n" + md_report)

    if args.markdown and args.markdown != "__STDOUT__":
        out_path = Path(args.markdown)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_report, encoding="utf-8")
        print(f"Markdown report written to {out_path}")

    if args.latex:
        latex_blocks = []
        for metric in results["metrics"]:
            s_tex = generate_demsar_all_vs_all_latex_table(
                results, metric=metric, dataset_label=ds_label
            )
            d_tex = generate_pairwise_delta_latex_matrix(
                results, metric=metric, dataset_label=ds_label
            )
            latex_blocks.extend([s_tex, "\n", d_tex, "\n"])
        full_latex = "\n".join(latex_blocks)
        if args.latex == "__STDOUT__":
            print("\n" + full_latex)
        else:
            out_path = Path(args.latex)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(full_latex, encoding="utf-8")
            print(f"LaTeX tables written to {out_path}")

    if args.save_tables:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        ds_slug = "_".join(args.dataset).lower().replace(" ", "_")
        cond_slug = args.condition.lower().replace(" ", "_")

        md_file = TABLES_DIR / f"demsar_all_vs_all_{ds_slug}_{cond_slug}.md"
        md_file.write_text(md_report, encoding="utf-8")
        print(f" - Markdown Report: {md_file}")

        latex_blocks = []
        for metric in results["metrics"]:
            s_tex = generate_demsar_all_vs_all_latex_table(
                results, metric=metric, dataset_label=ds_label
            )
            d_tex = generate_pairwise_delta_latex_matrix(
                results, metric=metric, dataset_label=ds_label
            )
            latex_blocks.extend([s_tex, "\n", d_tex, "\n"])
        tex_file = TABLES_DIR / f"demsar_all_vs_all_{ds_slug}_{cond_slug}.tex"
        tex_file.write_text("\n".join(latex_blocks), encoding="utf-8")
        print(f" - LaTeX Tables:    {tex_file}")


if __name__ == "__main__":
    main()
