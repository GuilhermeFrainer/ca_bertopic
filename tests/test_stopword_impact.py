import polars as pl

from src.make_table import generate_stopword_impact_latex_table
from src.results_analysis import compute_stopword_impact


def test_compute_stopword_impact_basic():
    # 2 runs for model_A, 2 runs for model_B
    df_std = pl.DataFrame(
        {
            "model_name": [
                "baseline_1_seed100",
                "baseline_2_seed200",
                "aligned_umap_1_seed100",
                "aligned_umap_2_seed200",
            ],
            "dataset_name": ["fed", "fed", "fed", "fed"],
            "random_state": [100, 200, 100, 200],
            "n_topics": [10, 20, 10, 20],
            "clustering_algo": ["hdbscan", "hdbscan", "hdbscan", "hdbscan"],
            "dim_red_algo": ["umap", "umap", "umap", "umap"],
            "c_v": [0.60, 0.70, 0.55, 0.65],
            "u_mass": [-1.0, -1.2, -0.8, -0.9],
        }
    )

    df_no_stop = pl.DataFrame(
        {
            "model_name": [
                "baseline_1_seed100",
                "baseline_2_seed200",
                "aligned_umap_1_seed100",
                "aligned_umap_2_seed200",
            ],
            "dataset_name": ["fed", "fed", "fed", "fed"],
            "random_state": [100, 200, 100, 200],
            "n_topics": [10, 20, 10, 20],
            "clustering_algo": ["hdbscan", "hdbscan", "hdbscan", "hdbscan"],
            "dim_red_algo": ["umap", "umap", "umap", "umap"],
            "c_v": [
                0.50,
                0.60,
                0.60,
                0.70,
            ],  # delta c_v: baseline=+0.10, aligned_umap=-0.05
            "u_mass": [
                -1.2,
                -1.5,
                -0.7,
                -0.8,
            ],  # delta u_mass: baseline=+0.20, +0.30 -> mean +0.25; aligned_umap=-0.10, -0.10 -> mean -0.10
        }
    )

    results = compute_stopword_impact(df_std, df_no_stop, dataset="fed")

    assert "c_v" in results
    assert "u_mass" in results

    cv_res = results["c_v"]
    base_match = cv_res.filter(pl.col("model_type") == "baseline")
    aligned_match = cv_res.filter(pl.col("model_type") == "aligned_umap")

    assert not base_match.is_empty()
    assert not aligned_match.is_empty()

    assert abs(base_match["mean_delta"][0] - 0.10) < 1e-5
    assert abs(aligned_match["mean_delta"][0] - (-0.05)) < 1e-5


def test_compute_stopword_impact_empty():
    empty_df = pl.DataFrame()
    res = compute_stopword_impact(empty_df, empty_df, dataset="fed")
    assert res == {}


def test_compute_stopword_impact_unmatched():
    df_std = pl.DataFrame(
        {
            "model_name": ["baseline_1_seed100"],
            "dataset_name": ["fed"],
            "c_v": [0.60],
        }
    )
    df_no_stop = pl.DataFrame(
        {
            "model_name": ["baseline_1_seed999"],  # non-matching seed/name
            "dataset_name": ["fed"],
            "c_v": [0.50],
        }
    )
    res = compute_stopword_impact(df_std, df_no_stop, dataset="fed")
    assert res == {}


def test_generate_stopword_impact_latex_table_coloring():
    # Construct synthetic results dict
    c_v_df = pl.DataFrame(
        {
            "model_type": ["baseline", "aligned_umap"],
            "mean_delta": [0.050, -0.025],
            "std_delta": [0.010, 0.005],
            "n_pairs": [2, 2],
        }
    )
    results = {"c_v": c_v_df}

    latex_table = generate_stopword_impact_latex_table(
        results, dataset="fed", pos_color="D4EDDA", neg_color="F8D7DA"
    )

    # Check LaTeX table structure and cell background colors
    assert "\\begin{table}" in latex_table
    assert (
        "\\cellcolor[HTML]{D4EDDA}" in latex_table
    )  # Green for positive delta (+0.050)
    assert "\\cellcolor[HTML]{F8D7DA}" in latex_table  # Red for negative delta (-0.025)
    assert "$+0.050" in latex_table
    assert "$-0.025" in latex_table
    assert "comparing representation stopwords removed vs. kept" in latex_table
