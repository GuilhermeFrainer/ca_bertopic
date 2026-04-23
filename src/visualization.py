from pathlib import Path

import pandas as pd
import plotly.express as px
import polars as pl

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
