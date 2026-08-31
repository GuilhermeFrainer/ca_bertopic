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

from src.results_analysis import calculate_hdbscan_noise_coverage

RESULTS_DIR = PROJECT_ROOT / "results"


def generate_latex_table(df: pl.DataFrame, result_type: str | None = None) -> str:
    """Generates a LaTeX table string for noise-cluster coverage.

    Args:
        df: Polars DataFrame containing noise coverage data.
        result_type: Optional result type identifier used to append preprocessing
            context to the table caption. Valid options:
            - 'standard': Standard unstemmed text with representation stopwords removed.
            - 'stemmed': Stemmed text with stopwords removed.
            - 'no_stopword_removal' (or 'with_stopwords', 'no_stopword'): Unstemmed text
              without representation stopword removal.

    Returns:
        A LaTeX table string.
    """
    res_descr = ""
    if result_type:
        rt = result_type.lower()
        if rt == "standard":
            res_descr = (
                " for Standard Unstemmed Text with Representation Stopwords Removed"
            )
        elif rt == "stemmed":
            res_descr = " for Stemmed Text with Stopwords Removed"
        elif rt in ("no_stopword", "no_stopword_removal", "with_stopwords"):
            res_descr = " for Unstemmed Text without Representation Stopword Removal"

    caption_text = (
        f"HDBSCAN Noise-Cluster Coverage across Random Seeds{res_descr} "
        r"(Mean $\pm$ Standard Deviation)"
    )
    col_header = (
        r"    Dataset & Model & Runs & Mean Noise Docs & "
        r"Mean Noise Coverage (\% $\pm$ SD) \\"
    )
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption_text}}}",
        r"\label{tab:noise_coverage}",
        r"\begin{tabular}{llrrr}",
        r"    \toprule",
        col_header,
        r"    \midrule",
    ]

    for row in df.iter_rows(named=True):
        dataset = row.get("dataset_name", "N/A")
        model = str(row.get("model_type", row.get("model_name", "N/A"))).replace(
            "_", r"\_"
        )
        runs = row.get("n_runs", 1)
        mean_outliers = row.get("outliers_mean", row.get("outliers", 0))
        mean_pct = row.get(
            "noise_coverage_pct_mean", row.get("noise_coverage_pct", 0.0)
        )
        std_pct = row.get("noise_coverage_pct_std", 0.0)

        if "noise_coverage_pct_std" in row and std_pct > 0:
            pct_str = f"${mean_pct:.2f} \\pm {std_pct:.2f}$\\%"
        else:
            pct_str = f"{mean_pct:.2f}\\%"

        lines.append(
            f"    {dataset} & {model} & {runs} & {mean_outliers:.1f} & {pct_str} \\\\"
        )

    lines.extend(
        [
            "    \\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate HDBSCAN noise-cluster coverage for topic models."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory containing results CSV files (default: results/)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a specific results CSV file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter results by dataset name (e.g., fed, yelp)",
    )
    parser.add_argument(
        "--result-type",
        type=str,
        choices=[
            "standard",
            "stemmed",
            "no_stopword_removal",
            "with_stopwords",
            "no_stopword",
            "all",
        ],
        default="all",
        help="Filter results by preprocessing result type (default: all)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed results per run instead of aggregating by model type",
    )
    parser.add_argument(
        "--merge-info0",
        action="store_true",
        help="Merge info0 model variants into base model types",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save results as CSV",
    )
    parser.add_argument(
        "--output-latex",
        type=str,
        default=None,
        help="Optional path to save results as LaTeX table",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    args = parser.parse_args()

    # Load CSV files
    dfs = []
    if args.input:
        input_path = Path(args.input)
        if input_path.exists():
            dfs.append(pl.read_csv(input_path))
        else:
            if not args.quiet:
                print(f"Error: Specified input file not found: {args.input}")
            sys.exit(1)
    else:
        results_path = Path(args.results_dir)
        csv_files = list(results_path.glob("*.csv"))
        if not csv_files:
            if not args.quiet:
                print(f"No CSV files found in directory: {args.results_dir}")
            sys.exit(1)

        target_res_type = args.result_type
        if target_res_type in ("no_stopword", "with_stopwords"):
            target_res_type = "no_stopword_removal"

        for csv_file in csv_files:
            filename = csv_file.name.lower()
            if target_res_type == "stemmed" and "stemmed" not in filename:
                continue
            if (
                target_res_type == "no_stopword_removal"
                and "no_stopword" not in filename
            ):
                continue
            if target_res_type == "standard" and (
                "stemmed" in filename or "no_stopword" in filename
            ):
                continue

            try:
                dfs.append(pl.read_csv(csv_file))
            except Exception as e:
                if not args.quiet:
                    print(f"Warning: Could not read {csv_file}: {e}")

    if not dfs:
        if not args.quiet:
            print("Error: No valid result DataFrames loaded.")
        sys.exit(1)

    combined_df = pl.concat(dfs, how="diagonal")

    coverage_df = calculate_hdbscan_noise_coverage(
        combined_df,
        dataset=args.dataset,
        group_by_model_type=not args.detailed,
        merge_info0=args.merge_info0,
    )

    if coverage_df.is_empty():
        if not args.quiet:
            print("No HDBSCAN model entries found matching criteria.")
        return

    has_output_file = bool(args.output_csv or args.output_latex)
    if not args.quiet and not has_output_file:
        # Set ASCII table rendering for windows terminals compatibility
        pl.Config.set_ascii_tables()
        pl.Config.set_tbl_rows(100)

        print("\n" + "=" * 80)
        print(" HDBSCAN NOISE-CLUSTER COVERAGE SUMMARY ")
        print("=" * 80)
        print(coverage_df)
        print("=" * 80 + "\n")

    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_df.write_csv(output_csv_path)
        if not args.quiet:
            print(f"Saved CSV report to: {output_csv_path}")

    if args.output_latex:
        latex_str = generate_latex_table(coverage_df, result_type=args.result_type)
        output_latex_path = Path(args.output_latex)
        output_latex_path.parent.mkdir(parents=True, exist_ok=True)
        output_latex_path.write_text(latex_str, encoding="utf-8")
        if not args.quiet:
            print(f"Saved LaTeX table to: {output_latex_path}")


if __name__ == "__main__":
    main()
