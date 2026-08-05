import sys
from pathlib import Path

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# noqa: E402
from src.results_analysis import calculate_hdbscan_noise_coverage  # noqa: E402


def test_calculate_hdbscan_noise_coverage_aggregated():
    data = {
        "model_name": [
            "baseline_1_seed1",
            "baseline_1_seed2",
            "aligned_umap_1_seed1",
            "kmeans_model_1",
        ],
        "dataset_name": ["fed", "fed", "fed", "fed"],
        "clustering_algo": ["hdbscan", "hdbscan", "hdbscan", "k_means"],
        "n_observations": [1000, 1000, 1000, 1000],
        "outliers": [200, 250, 100, 0],
    }
    df = pl.DataFrame(data)

    res = calculate_hdbscan_noise_coverage(df, group_by_model_type=True)

    # Should filter out k_means and keep baseline and aligned_umap
    assert res.height == 2
    assert "baseline" in res["model_type"].to_list()
    assert "aligned_umap" in res["model_type"].to_list()

    baseline_row = res.filter(pl.col("model_type") == "baseline")
    assert baseline_row["n_runs"][0] == 2
    assert baseline_row["outliers_mean"][0] == 225.0
    assert baseline_row["noise_coverage_pct_mean"][0] == 22.5
    assert baseline_row["clustered_coverage_pct_mean"][0] == 77.5


def test_calculate_hdbscan_noise_coverage_detailed():
    data = {
        "model_name": ["baseline_1_seed1", "aligned_umap_1_seed1"],
        "dataset_name": ["fed", "fed"],
        "clustering_algo": ["hdbscan", "hdbscan"],
        "n_observations": [500, 500],
        "outliers": [50, 100],
    }
    df = pl.DataFrame(data)

    res = calculate_hdbscan_noise_coverage(df, group_by_model_type=False)

    assert res.height == 2
    assert "noise_ratio" in res.columns
    assert "noise_coverage_pct" in res.columns
    assert "clustered_coverage_pct" in res.columns
    base_val = res.filter(pl.col("model_name") == "baseline_1_seed1")[
        "noise_coverage_pct"
    ][0]
    assert base_val == 10.0
    aligned_val = res.filter(pl.col("model_name") == "aligned_umap_1_seed1")[
        "noise_coverage_pct"
    ][0]
    assert aligned_val == 20.0
