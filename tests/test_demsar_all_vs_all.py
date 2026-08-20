import polars as pl
import pytest

from src.make_table import (
    generate_demsar_all_vs_all_latex_table,
    generate_demsar_all_vs_all_markdown_table,
    generate_demsar_all_vs_all_report,
    generate_pairwise_delta_latex_matrix,
    generate_pairwise_delta_markdown_matrix,
)
from src.results_analysis import (
    compute_demsar_all_vs_all,
    compute_nemenyi_cliques,
    friedman_omnibus_test,
    nemenyi_critical_difference,
    pairwise_all_vs_all_tests,
)


def test_friedman_omnibus_known_case():
    # 3 models, 4 blocks:
    # Model 1 always rank 1, Model 2 always rank 2, Model 3 always rank 3
    mean_ranks = [1.0, 2.0, 3.0]
    n_blocks = 4
    k_models = 3

    res = friedman_omnibus_test(mean_ranks, n_blocks=n_blocks, k_models=k_models)

    # sum(R_j^2) = 1 + 4 + 9 = 14
    # term = 3 * 16 / 4 = 12
    # chi2 = (12 * 4 / 12) * (14 - 12) = 4 * 2 = 8.0
    assert abs(res["chi2_f"] - 8.0) < 1e-5
    assert res["df1"] == 2
    assert res["df2"] == 6
    # Denominator in Iman-Davenport = 4 * 2 - 8.0 = 0.0 -> inf
    assert res["f_f"] == float("inf")
    assert res["p_f_f"] == 0.0


def test_friedman_omnibus_no_difference():
    # All models have identical average rank: (k+1)/2 = 2.0
    mean_ranks = [2.0, 2.0, 2.0]
    res = friedman_omnibus_test(mean_ranks, n_blocks=5, k_models=3)
    assert abs(res["chi2_f"] - 0.0) < 1e-5
    assert abs(res["f_f"] - 0.0) < 1e-5
    assert abs(res["p_chi2"] - 1.0) < 1e-5
    assert abs(res["p_f_f"] - 1.0) < 1e-5


def test_nemenyi_critical_difference():
    # Demšar (2006) Table 5:
    # For k=2, q_0.05 = 1.960. With N=6:
    # CD = 1.960 * sqrt(2*3 / 36) = 1.960 * sqrt(1/6) = 0.800
    cd_k2 = nemenyi_critical_difference(n_blocks=6, k_models=2, alpha=0.05)
    assert pytest.approx(cd_k2, rel=1e-2) == 0.800

    # For k=3, q_0.05 = 2.344. With N=10:
    # CD = 2.344 * sqrt(3*4 / 60) = 2.344 * sqrt(0.2) = 1.048
    cd_k3 = nemenyi_critical_difference(n_blocks=10, k_models=3, alpha=0.05)
    assert pytest.approx(cd_k3, rel=1e-2) == 1.048

    # Edge cases
    assert nemenyi_critical_difference(n_blocks=0, k_models=3) == 0.0
    assert nemenyi_critical_difference(n_blocks=5, k_models=1) == 0.0


def test_compute_nemenyi_cliques():
    # 4 models: M1=1.0, M2=1.5, M3=2.8, M4=3.5. CD = 1.0
    models = ["M1", "M2", "M3", "M4"]
    mean_ranks = {"M1": 1.0, "M2": 1.5, "M3": 2.8, "M4": 3.5}
    cd = 1.0

    # Range [M1, M2] diff=0.5 <= 1.0 -> clique 'a'
    # Range [M3, M4] diff=0.7 <= 1.0 -> clique 'b'
    # Range [M2, M3] diff=1.3 > 1.0 -> not a clique
    cliques = compute_nemenyi_cliques(models, mean_ranks, cd=cd)
    assert cliques["M1"] == "a"
    assert cliques["M2"] == "a"
    assert cliques["M3"] == "b"
    assert cliques["M4"] == "b"

    # Overlapping cliques: CD = 1.5
    # [M1..M2] diff=0.5, [M2..M3] diff=1.3 <= 1.5, [M3..M4] diff=0.7 <= 1.5
    # Maximal: [M1, M2], [M2, M3], [M3, M4]
    cliques_overlap = compute_nemenyi_cliques(models, mean_ranks, cd=1.5)
    assert "a" in cliques_overlap["M1"]
    assert "M3" in cliques_overlap


def test_pairwise_all_vs_all_tests():
    models = ["M1", "M2", "M3"]
    mean_ranks = {"M1": 1.0, "M2": 2.0, "M3": 3.0}
    n_blocks = 10
    k_models = 3

    df_pairs = pairwise_all_vs_all_tests(
        models=models,
        mean_ranks=mean_ranks,
        n_blocks=n_blocks,
        k_models=k_models,
        alpha=0.05,
        correction="holm",
    )

    assert not df_pairs.is_empty()
    assert len(df_pairs) == 6  # 3 * 2 ordered pairs

    # Check pair M1 vs M3
    m1_m3 = df_pairs.filter((pl.col("model_a") == "M1") & (pl.col("model_b") == "M3"))
    assert not m1_m3.is_empty()
    assert m1_m3["rank_diff"][0] == -2.0
    assert m1_m3["p_raw"][0] < 0.01
    assert m1_m3["p_adj"][0] < 0.05
    assert m1_m3["is_significant"][0] is True


def test_compute_demsar_all_vs_all_synthetic():
    # Build synthetic multi-model experiment results:
    # 3 models: 'baseline', 'mv_spectral', 'aligned_umap'
    # 5 topic counts (1..5), 3 random seeds (1, 2, 3)
    rows = []
    for t in range(1, 6):
        for s in [1, 2, 3]:
            # aligned_umap is clearly best, baseline is mid, mv_spectral is lowest
            rows.append(
                {
                    "model_name": f"aligned_umap_{t}_seed{s}",
                    "dataset_name": "fed",
                    "random_state": s,
                    "clustering_algo": "hdbscan",
                    "dim_red_algo": "umap",
                    "c_v": 0.80 + 0.01 * t + 0.001 * s,
                    "u_mass": -0.50 + 0.01 * t,
                    "c_npmi": 0.20,
                    "irbo": 0.90,
                    "topic_diversity": 0.85,
                }
            )
            rows.append(
                {
                    "model_name": f"baseline_{t}_seed{s}",
                    "dataset_name": "fed",
                    "random_state": s,
                    "clustering_algo": "hdbscan",
                    "dim_red_algo": "umap",
                    "c_v": 0.70 + 0.01 * t + 0.001 * s,
                    "u_mass": -0.80 + 0.01 * t,
                    "c_npmi": 0.15,
                    "irbo": 0.80,
                    "topic_diversity": 0.75,
                }
            )
            rows.append(
                {
                    "model_name": f"mv_spectral_{t}_seed{s}",
                    "dataset_name": "fed",
                    "random_state": s,
                    "clustering_algo": "spectral",
                    "dim_red_algo": "umap",
                    "c_v": 0.50 + 0.01 * t + 0.001 * s,
                    "u_mass": -1.20 + 0.01 * t,
                    "c_npmi": 0.05,
                    "irbo": 0.70,
                    "topic_diversity": 0.65,
                }
            )

    df_synth = pl.DataFrame(rows)

    results = compute_demsar_all_vs_all(
        df=df_synth,
        dataset="fed",
        alpha=0.05,
    )

    assert "metrics" in results
    assert "c_v" in results["metrics"]
    cv_res = results["metrics"]["c_v"]

    assert cv_res["n_blocks"] == 5
    assert cv_res["k_models"] == 3
    assert cv_res["is_significant"] is True

    # Summary table checks
    summary = cv_res["summary_table"]
    assert not summary.is_empty()
    assert summary["Model"][0] == "aligned_umap"
    assert summary["Mean Rank"][0] == 1.0
    assert summary["Is Best"][0] is True

    # Delta matrix checks
    delta_mat = cv_res["pairwise_delta_matrix"]
    assert not delta_mat.is_empty()
    assert delta_mat.columns == ["Model", "aligned_umap", "baseline", "mv_spectral"]


def test_demsar_all_vs_all_table_generators():
    # Test rendering to Markdown, LaTeX, and full report
    rows = []
    for t in range(1, 6):
        for s in [10, 20, 30]:
            rows.append(
                {
                    "model_name": f"aligned_umap_{t}_seed{s}",
                    "dataset_name": "fed",
                    "random_state": s,
                    "clustering_algo": "hdbscan",
                    "dim_red_algo": "umap",
                    "c_v": 0.80 + 0.01 * t,
                    "u_mass": -0.50,
                    "c_npmi": 0.20,
                    "irbo": 0.90,
                    "topic_diversity": 0.85,
                }
            )
            rows.append(
                {
                    "model_name": f"baseline_{t}_seed{s}",
                    "dataset_name": "fed",
                    "random_state": s,
                    "clustering_algo": "hdbscan",
                    "dim_red_algo": "umap",
                    "c_v": 0.60 + 0.01 * t,
                    "u_mass": -1.00,
                    "c_npmi": 0.10,
                    "irbo": 0.80,
                    "topic_diversity": 0.70,
                }
            )

    df_synth = pl.DataFrame(rows)
    results = compute_demsar_all_vs_all(df_synth, dataset="fed", alpha=0.05)

    md_table = generate_demsar_all_vs_all_markdown_table(
        results, metric="c_v", dataset_label="FED"
    )
    assert "Demšar All-vs-All Ranking Summary: Topic Coherence (C_V) [FED]" in md_table
    assert "**aligned_umap**" in md_table

    md_delta = generate_pairwise_delta_markdown_matrix(
        results, metric="c_v", dataset_label="FED"
    )
    assert "Pairwise Delta Matrix: C_V [FED]" in md_delta
    assert "**aligned_umap**" in md_delta

    tex_table = generate_demsar_all_vs_all_latex_table(
        results, metric="c_v", dataset_label="FED"
    )
    assert "\\begin{table}" in tex_table
    assert "\\caption{Demšar (2006) All-vs-All Ranking Summary" in tex_table

    tex_matrix = generate_pairwise_delta_latex_matrix(
        results, metric="c_v", dataset_label="FED"
    )
    assert "\\begin{table}" in tex_matrix
    assert "\\caption{Demšar (2006) Pairwise Delta Matrix" in tex_matrix

    full_report = generate_demsar_all_vs_all_report(results, dataset_label="FED")
    assert "# Demšar (2006) All-vs-All Statistical Comparison Report" in full_report
