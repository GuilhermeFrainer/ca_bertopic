import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.make_table import (  # noqa: E402
    generate_best_models_markdown_table,
    generate_best_models_table_data,
    generate_noise_coverage_latex_table,
    generate_noise_coverage_markdown_table,
    generate_stopword_impact_markdown_table,
    generate_stopword_impact_table_data,
    style_demsar_delta_dataframe,
    style_demsar_pairwise_matrix,
)
from src.results_analysis import find_best_models  # noqa: E402


@pytest.fixture
def sample_best_models_results():
    data = {
        "model_name": ["baseline_1", "baseline_2", "mv_spectral_1", "mv_spectral_2"],
        "dataset_name": ["fed", "fed", "fed", "fed"],
        "c_v": [0.55, 0.72, 0.61, 0.83],
        "u_mass": [-1.5, -1.2, -1.4, -1.1],
        "irbo": [0.3, 0.5, 0.4, 0.6],
        "n_topics": [10, 20, 10, 20],
        "random_state": [1, 2, 1, 2],
    }
    df = pl.DataFrame(data)
    return find_best_models(df, "fed", average=True)


def test_generate_best_models_table_data(sample_best_models_results):
    data = generate_best_models_table_data(
        sample_best_models_results, dump=False, average=True
    )

    display_df = data["display_df"]
    numeric_df = data["numeric_df"]
    styler = data["styler"]
    metric_cols = data["metric_cols"]

    assert not display_df.empty
    assert not numeric_df.empty
    assert styler is not None
    assert "Model" in display_df.columns
    assert "C_v" in metric_cols
    assert "UMass" in metric_cols

    # Check that display_df contains formatted strings with ±
    cv_val = display_df["C_v"].iloc[0]
    assert "±" in cv_val or isinstance(float(cv_val), float)

    # Check that numeric_df contains valid float
    assert isinstance(numeric_df["C_v"].iloc[0], (float, int))


def test_generate_best_models_markdown_table(sample_best_models_results):
    md = generate_best_models_markdown_table(
        sample_best_models_results, dataset="fed", average=True
    )
    assert "###" in md
    assert "| Model |" in md
    assert "C_v" in md
    assert "UMass" in md
    # Should bold best model in markdown
    assert "**" in md


def test_generate_stopword_impact_table_data():
    results = {
        "c_v": pl.DataFrame(
            {
                "model_type": ["baseline", "mv_spectral"],
                "mean_delta": [0.045, -0.012],
                "std_delta": [0.010, 0.005],
                "n_pairs": [5, 5],
            }
        )
    }

    data = generate_stopword_impact_table_data(results)
    display_df = data["display_df"]
    assert not display_df.empty
    assert "Δ C_v" in display_df.columns
    # Check + sign for positive delta
    assert any("+" in str(v) for v in display_df["Δ C_v"])

    md = generate_stopword_impact_markdown_table(results, dataset="fed")
    assert "### Representation Stopword Impact" in md
    assert "| Model |" in md


def test_generate_noise_coverage_tables():
    noise_df = pl.DataFrame(
        {
            "dataset_name": ["fed", "fed"],
            "model_type": ["baseline", "mv_spectral"],
            "n_runs": [15, 15],
            "outliers_mean": [1023.3, 850.0],
            "noise_coverage_pct_mean": [18.79, 15.60],
            "noise_coverage_pct_std": [0.85, 0.50],
        }
    )

    latex = generate_noise_coverage_latex_table(noise_df, result_type="standard")
    assert "\\begin{table}" in latex
    assert "\\caption" in latex
    assert "HDBSCAN Noise-Cluster Coverage" in latex
    assert "18.79" in latex

    md = generate_noise_coverage_markdown_table(noise_df)
    assert "### HDBSCAN Noise-Cluster Coverage" in md
    assert "| Dataset | Model |" in md
    assert "18.79%" in md


def test_style_demsar_delta_dataframe():
    df = pd.DataFrame(
        {
            "Model": ["BERTopic₁", "CAST₁"],
            "UMass": ["+0.051 ± 0.012*", "-0.020 ± 0.008"],
            "C_v": ["-0.010 ± 0.005", "+0.033 ± 0.011*"],
        }
    )
    styler = style_demsar_delta_dataframe(df)
    assert styler is not None
    # Verify HTML rendering does not throw errors
    html = styler.to_html()
    assert "#D4EDDA" in html  # green for +
    assert "#F8D7DA" in html  # red for -


def test_style_demsar_pairwise_matrix():
    df = pd.DataFrame(
        {
            "Model": ["CAST₁", "BERTopic₁"],
            "CAST₁": ["-", "-0.040*"],
            "BERTopic₁": ["+0.040*", "-"],
        }
    )
    styler = style_demsar_pairwise_matrix(df)
    assert styler is not None
    html = styler.to_html()
    assert "#D4EDDA" in html  # green for +
