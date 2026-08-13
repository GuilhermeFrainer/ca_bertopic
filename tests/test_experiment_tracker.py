from pathlib import Path

import polars as pl
import pytest

from src.experiment_tracker import (
    build_coverage_matrix,
    classify_result_condition,
    scan_experiment_configs,
)


@pytest.fixture
def mock_exp_dir(tmp_path: Path) -> Path:
    """Creates a temporary experiments directory layout for testing."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()

    # Base datasets dir (should be ignored)
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()
    (ds_dir / "fed.yaml").write_text("dataset: fed")

    # Active experiment dirs
    fed_dir = exp_dir / "fed"
    fed_dir.mkdir()
    (fed_dir / "fed_standard_aligned_umap.yaml").write_text("experiment: name")
    (fed_dir / "fed_standard_baseline.yaml").write_text("experiment: name")

    fed_stemmed_dir = exp_dir / "fed_stemmed"
    fed_stemmed_dir.mkdir()
    (fed_stemmed_dir / "fed_standard_aligned_umap.yaml").write_text("experiment: name")

    # Archive dir
    arch_dir = exp_dir / "archive" / "fed"
    arch_dir.mkdir(parents=True)
    (arch_dir / "fed_opt_legacy.yaml").write_text("experiment: name")

    return exp_dir


def test_scan_experiment_configs(mock_exp_dir: Path):
    # Test active experiments only (default)
    configs = scan_experiment_configs(mock_exp_dir, include_archived=False)

    stems = [c["stem"] for c in configs]
    assert "fed_standard_aligned_umap" in stems
    assert "fed_standard_baseline" in stems
    assert "fed" not in stems  # datasets/fed.yaml excluded
    assert "fed_opt_legacy" not in stems  # archive excluded

    datasets = set(c["dataset_label"] for c in configs)
    assert datasets == {"fed"}

    # Test with archived included
    configs_with_arch = scan_experiment_configs(mock_exp_dir, include_archived=True)
    stems_with_arch = [c["stem"] for c in configs_with_arch]
    assert "fed_opt_legacy" in stems_with_arch


def test_classify_result_condition():
    # Explicit column tests
    assert (
        classify_result_condition("file.csv", stopword_removal_col="keep_rep_stopwords")
        == "keep_rep_stopwords"
    )
    assert (
        classify_result_condition(
            "file.csv", stopword_removal_col="remove_rep_stopwords"
        )
        == "remove_rep_stopwords"
    )
    assert (
        classify_result_condition("file.csv", stopword_removal_col="stemmed")
        == "stemmed"
    )

    # Filename fallback tests
    assert classify_result_condition("fed_stemmed_aligned_umap.csv") == "stemmed"
    assert (
        classify_result_condition("fed_keep_rep_stopwords_aligned_umap.csv")
        == "keep_rep_stopwords"
    )
    assert (
        classify_result_condition("fed_standard_aligned_umap-20260803.csv")
        == "remove_rep_stopwords"
    )


def test_build_coverage_matrix(mock_exp_dir: Path):
    experiments = scan_experiment_configs(mock_exp_dir, include_archived=False)

    # Mock results dataframe
    results_df = pl.DataFrame(
        {
            "source_file": [
                "fed_standard_aligned_umap-20260803-120000-123.csv",
                "fed_stemmed_standard_aligned_umap-20260805-120000-123.csv",
                "fed_standard_aligned_umap_dry_run_100_keep_rep_stopwords-20260806.csv",
            ],
            "experiment_id": [
                "fed_standard_aligned_umap",
                "fed_stemmed_standard_aligned_umap",
                "fed_standard_aligned_umap_dry_run_100",
            ],
            "dataset_label": ["fed", "fed", "fed"],
            "stopword_removal": [
                "remove_rep_stopwords",
                "stemmed",
                "keep_rep_stopwords",
            ],
        }
    )

    matrix = build_coverage_matrix(experiments, results_df)

    assert not matrix.is_empty()
    assert "dataset_label" in matrix.columns
    assert "coverage_score" in matrix.columns

    aligned_row = matrix.filter(
        pl.col("experiment_name") == "fed_standard_aligned_umap"
    ).to_dicts()[0]

    assert aligned_row["remove_rep_stopwords"].startswith("✅ Done")
    assert aligned_row["stemmed"].startswith("✅ Done")
    assert aligned_row["keep_rep_stopwords"].startswith("⚠️ Dry Run")
    assert aligned_row["coverage_score"] == "3/3"
    assert aligned_row["coverage_status"] == "Fully Completed"

    baseline_row = matrix.filter(
        pl.col("experiment_name") == "fed_standard_baseline"
    ).to_dicts()[0]
    assert baseline_row["coverage_score"] == "0/3"
    assert baseline_row["coverage_status"] == "Not Run"
