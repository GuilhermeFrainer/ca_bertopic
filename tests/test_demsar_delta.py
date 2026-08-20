import polars as pl
import pytest

from src.make_table import (
    generate_demsar_delta_latex_table,
    generate_demsar_delta_markdown_table,
)
from src.results_analysis import (
    compute_demsar_delta_table,
    holm_bonferroni,
    parse_model_type_and_topic,
    wilcoxon_exact_test,
)


def test_wilcoxon_exact_small_sample():
    # N=5 strictly positive differences
    diffs = [0.1, 0.2, 0.3, 0.4, 0.5]
    stat, p_two = wilcoxon_exact_test(diffs, alternative="two-sided")
    assert stat == 0.0
    assert abs(p_two - 0.0625) < 1e-5

    stat, p_greater = wilcoxon_exact_test(diffs, alternative="greater")
    assert stat == 15.0
    assert abs(p_greater - 0.03125) < 1e-5

    # All zeros
    zeros = [0.0, 0.0, 0.0, 0.0, 0.0]
    stat, p_zero = wilcoxon_exact_test(zeros, alternative="two-sided")
    assert stat == 0.0
    assert p_zero == 1.0

    # Empty
    stat, p_empty = wilcoxon_exact_test([], alternative="two-sided")
    assert p_empty == 1.0


def test_holm_bonferroni_adjustment():
    # Empty
    assert holm_bonferroni([]) == []

    # Single p-value
    assert holm_bonferroni([0.04]) == [0.04]

    # Multiple identical p-values: [0.0625, 0.0625] -> [0.125, 0.125]
    res = holm_bonferroni([0.0625, 0.0625])
    assert pytest.approx(res, rel=1e-4) == [0.125, 0.125]

    # Increasing sequence
    raw_p = [0.01, 0.04, 0.05]
    # step 1: 0.01 * 3 = 0.03
    # step 2: max(0.03, 0.04 * 2) = 0.08
    # step 3: max(0.08, 0.05 * 1) = 0.08
    adj = holm_bonferroni(raw_p)
    assert pytest.approx(adj, rel=1e-4) == [0.03, 0.08, 0.08]

    # Monotonicity check
    raw_unordered = [0.05, 0.01, 0.04]
    adj_unordered = holm_bonferroni(raw_unordered)
    assert pytest.approx(adj_unordered, rel=1e-4) == [0.08, 0.03, 0.08]


def test_parse_model_type_and_topic():
    # Standard format with topic count index and seed
    mtype, topic_idx = parse_model_type_and_topic("baseline_1_seed36201624")
    assert mtype == "baseline"
    assert topic_idx == 1

    # Stemmed prefix
    mtype, topic_idx = parse_model_type_and_topic("stemmed_mv_spectral_3_seed57116123")
    assert mtype == "mv_spectral"
    assert topic_idx == 3

    # With info0 stripping
    mtype, topic_idx = parse_model_type_and_topic(
        "mv_co_reg_spectral_info0_5_seed62613654", merge_info0=True
    )
    assert mtype == "mv_co_reg_spectral"
    assert topic_idx == 5

    # Without info0 stripping
    mtype, topic_idx = parse_model_type_and_topic(
        "mv_co_reg_spectral_info0_5_seed62613654", merge_info0=False
    )
    assert mtype == "mv_co_reg_spectral_info0"
    assert topic_idx == 5


def test_compute_demsar_delta_table_synthetic():
    # Construct synthetic data: 1 model, 5 topic counts (1..5), 3 seeds (10, 20, 30)
    rows_default = []
    rows_alt = []

    for k in range(1, 6):
        for seed in [10, 20, 30]:
            # Baseline: c_v = 0.50 + 0.01*k
            rows_default.append(
                {
                    "model_name": f"baseline_{k}_seed{seed}",
                    "dataset_name": "fed",
                    "random_state": seed,
                    "clustering_algo": "hdbscan",
                    "dim_red_algo": "umap",
                    "c_v": 0.50 + 0.01 * k,
                    "u_mass": -1.0,
                    "c_npmi": 0.10,
                    "irbo": 0.80,
                    "topic_diversity": 0.70,
                }
            )
            # Alternative: c_v improves by exactly 0.05 for all topic counts
            rows_alt.append(
                {
                    "model_name": f"stemmed_baseline_{k}_seed{seed}",
                    "dataset_name": "fed",
                    "random_state": seed,
                    "clustering_algo": "hdbscan",
                    "dim_red_algo": "umap",
                    "c_v": 0.50 + 0.01 * k + 0.05,
                    "u_mass": -1.0,
                    "c_npmi": 0.10,
                    "irbo": 0.80,
                    "topic_diversity": 0.70,
                }
            )

    df_default = pl.DataFrame(rows_default)
    df_alt = pl.DataFrame(rows_alt)

    # Run Demšar evaluation with alpha=0.10, two-sided
    results = compute_demsar_delta_table(
        df_default=df_default,
        df_alternative=df_alt,
        dataset="fed",
        alpha=0.10,
        alternative="two-sided",
        correction="per_metric",
    )

    assert not results["df_summary"].is_empty()
    assert not results["df_details"].is_empty()
    assert results["models"] == ["baseline"]
    assert "c_v" in results["metrics"]

    details = results["df_details"]
    cv_row = details.filter(
        (pl.col("model_type") == "baseline") & (pl.col("metric") == "c_v")
    )
    assert not cv_row.is_empty()
    assert abs(cv_row["mean_delta"][0] - 0.05) < 1e-5
    # All 5 differences are +0.05 -> raw exact p-value is 0.0625
    assert abs(cv_row["p_raw"][0] - 0.0625) < 1e-5
    # Single model in family -> p_adj = 0.0625 < 0.10 -> significant!
    assert cv_row["is_significant"][0] is True

    # Summary table should format as +0.050*
    summary = results["df_summary"]
    assert summary.filter(pl.col("Model") == "baseline")["c_v"][0] == "+0.050*"


def test_compute_demsar_delta_table_empty():
    empty = pl.DataFrame()
    res = compute_demsar_delta_table(empty, empty, dataset="fed")
    assert res["df_summary"].is_empty()
    assert res["models"] == []


def test_table_renderers():
    # Synthetic results
    df_summary = pl.DataFrame(
        {
            "Model": ["baseline", "aligned_umap"],
            "c_v": ["+0.050*", "-0.020"],
            "u_mass": ["+0.200*", "+0.100"],
        }
    )
    df_details = pl.DataFrame(
        {
            "model_type": ["baseline", "baseline", "aligned_umap", "aligned_umap"],
            "metric": ["c_v", "u_mass", "c_v", "u_mass"],
            "mean_delta": [0.050, 0.200, -0.020, 0.100],
            "std_delta": [0.005, 0.010, 0.002, 0.008],
            "is_significant": [True, True, False, False],
        }
    )
    results_dict = {
        "df_summary": df_summary,
        "df_details": df_details,
        "models": ["baseline", "aligned_umap"],
        "metrics": ["c_v", "u_mass"],
        "alpha": 0.10,
        "correction": "per_metric",
    }

    # Markdown Table
    md = generate_demsar_delta_markdown_table(
        results_dict, dataset="fed", condition_name="Stemmed"
    )
    assert "Performance Delta Table: Stemmed vs. Default (FED)" in md
    assert "**baseline**" in md
    assert "+0.050*" in md

    # LaTeX Table
    tex = generate_demsar_delta_latex_table(
        results_dict, dataset="fed", condition_name="Stemmed"
    )
    assert "\\begin{table}" in tex
    assert "\\cellcolor[HTML]{D4EDDA}" in tex
    assert "tab:demsar_delta_fed_stemmed" in tex
