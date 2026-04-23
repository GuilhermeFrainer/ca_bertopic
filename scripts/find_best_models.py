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

RESULTS_DIR = PROJECT_ROOT / "results"

# Mapping for prettier metric names
METRIC_DISPLAY_MAP = {
    "u_mass": "C<sub>UMass</sub>",
    "c_v": "C<sub>v</sub>",
    "c_npmi": "C<sub>npmi</sub>",
    "irbo": "IRBO",
    "topic_diversity": "Topic Diversity",
}

# Mapping for prettier model names (Plotly version)
MODEL_RENAME_MAP = {
    "append_umap": "Naive",
    "mv_co_reg_spectral": "CA-BERTopic<sub>co-reg</sub>",
    "mv_co_reg_spectral_info0": "CA-BERTopic<sub>co-reg-info0</sub>",
    "baseline": "BERTopic",
    "umap_spectral": "BERTopic<sub>Spectral</sub>",
    "mv_spectral": "CA-BERTopic<sub>Spectral</sub>",
    "mv_spectral_info0": "CA-BERTopic<sub>Spectral-info0</sub>",
    "aligned_umap": "CA-BERTopic<sub>Aligned</sub>",
}


def prepare_plot_data(results: dict[str, pl.DataFrame], dump: bool):
    """Common data preparation for Plotly visualizations."""
    import pandas as pd

    id_col = "best_model_name" if dump else "model_type"
    rows = []
    for metric, metric_df in results.items():
        display_metric = METRIC_DISPLAY_MAP.get(metric, metric)
        for row in metric_df.iter_rows(named=True):
            model_id = row[id_col]
            model_type = row["model_type"]
            display_model = MODEL_RENAME_MAP.get(model_id, model_id.replace("_", " "))
            display_model_type = MODEL_RENAME_MAP.get(
                model_type, model_type.replace("_", " ")
            )
            rows.append(
                {
                    "Model": display_model,
                    "ModelType": display_model_type,
                    "Metric": display_metric,
                    "Value": row["max_value"],
                    "RawMetric": metric,
                }
            )

    df = pd.DataFrame(rows)

    # Apply Min-Max Normalization per Metric
    df["NormalizedValue"] = df.groupby("Metric")["Value"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 1.0
    )

    return df


def generate_star_plot(results: dict[str, pl.DataFrame], dump: bool, output_path: Path):
    """Generates a star plot (radar chart) using Plotly with Min-Max normalization."""
    import plotly.express as px

    df = prepare_plot_data(results, dump)

    # Build the star plot
    fig = px.line_polar(
        df,
        r="NormalizedValue",
        theta="Metric",
        color="Model",
        line_close=True,
        hover_data={
            "Value": ":.4f",
            "NormalizedValue": False,
            "Model": True,
            "Metric": True,
        },
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_traces(
        fill="toself",
        opacity=0.2,
        line=dict(width=3.0),
        marker=dict(size=8),
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor="#E5E5E5",
            ),
            angularaxis=dict(
                gridcolor="#E5E5E5",
                linecolor="#E5E5E5",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=20, l=40, r=40),
    )

    save_plotly_figure(fig, output_path, ".pdf")


def generate_parallel_plot(
    results: dict[str, pl.DataFrame], dump: bool, output_path: Path
):
    """Generates a parallel coordinates plot (lines) using Plotly."""
    import pandas as pd
    import plotly.express as px

    df = prepare_plot_data(results, dump)

    # Sort metrics for consistent X axis
    metric_order = [METRIC_DISPLAY_MAP.get(m, m) for m in METRIC_DISPLAY_MAP]
    df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
    df = df.sort_values("Metric")

    fig = px.line(
        df,
        x="Metric",
        y="NormalizedValue",
        color="ModelType",
        line_group="Model",
        hover_data={"Value": ":.4f", "Model": True, "NormalizedValue": False},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
        markers=True,
    )

    fig.update_traces(opacity=0.6, line=dict(width=2))

    fig.update_layout(
        yaxis=dict(title="Relative Performance (Normalized)", range=[0, 1.05]),
        xaxis=dict(title="Metric"),
        legend=dict(
            title="Model Type",
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=20, l=40, r=40),
    )

    save_plotly_figure(fig, output_path, "_parallel.pdf")


def generate_cleveland_plot(
    results: dict[str, pl.DataFrame], dump: bool, output_path: Path
):
    """Generates a Cleveland dot plot using Plotly."""
    import plotly.express as px

    df = prepare_plot_data(results, dump)

    fig = px.scatter(
        df,
        x="NormalizedValue",
        y="Metric",
        color="Model",
        hover_data={"Value": ":.4f", "Model": True, "NormalizedValue": False},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_traces(marker=dict(size=12, opacity=0.8))

    fig.update_layout(
        xaxis=dict(title="Relative Performance (Normalized)", range=[-0.05, 1.05]),
        yaxis=dict(title="Metric", automargin=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=20, l=40, r=40),
    )

    save_plotly_figure(fig, output_path, "_cleveland.pdf")


def save_plotly_figure(fig, output_path: Path, default_suffix: str):
    """Helper to save a Plotly figure to disk."""
    if not output_path.suffix:
        output_path = output_path.with_suffix(default_suffix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path))
    print(f"Plot saved to {output_path}")


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
    )

    if not results:
        print(f"No valid metric results found for dataset: {dataset}")
        return

    if args.star_plot:
        generate_star_plot(results, args.dump, Path(args.star_plot))

    if args.parallel:
        generate_parallel_plot(results, args.dump, Path(args.parallel))

    if args.cleveland:
        generate_cleveland_plot(results, args.dump, Path(args.cleveland))

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
