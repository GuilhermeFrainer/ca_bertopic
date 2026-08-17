from pathlib import Path

import polars as pl
import pytest

from src.experiment_tracker import (
    DEFAULT_SLURM_SCRIPT_CLUSTER,
    DEFAULT_SLURM_SCRIPT_LOCAL,
    build_coverage_matrix,
    classify_result_condition,
    extract_model_name,
    generate_grouped_slurm_commands,
    generate_slurm_command,
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


def test_build_coverage_matrix_with_nan_metrics(mock_exp_dir: Path):
    experiments = scan_experiment_configs(mock_exp_dir, include_archived=False)

    # Mock results dataframe where remove_rep_stopwords has NaN metrics
    results_df = pl.DataFrame(
        {
            "source_file": [
                "fed_standard_aligned_umap-20260803-120000-123.csv",
                "fed_stemmed_standard_aligned_umap-20260805-120000-123.csv",
            ],
            "experiment_id": [
                "fed_standard_aligned_umap",
                "fed_stemmed_standard_aligned_umap",
            ],
            "dataset_label": ["fed", "fed"],
            "stopword_removal": [
                "remove_rep_stopwords",
                "stemmed",
            ],
            "u_mass": [float("nan"), -2.5],
            "c_v": [None, 0.65],
        }
    )

    matrix = build_coverage_matrix(experiments, results_df)
    aligned_row = matrix.filter(
        pl.col("experiment_name") == "fed_standard_aligned_umap"
    ).to_dicts()[0]

    assert aligned_row["remove_rep_stopwords"].startswith("❌ Error")
    assert "NaNs" in aligned_row["remove_rep_stopwords"]
    assert aligned_row["stemmed"].startswith("✅ Done")
    assert aligned_row["coverage_status"] == "Partially Completed"
    assert aligned_row["coverage_score"] == "1/3"


def test_build_coverage_matrix_with_partial_nan_metrics(mock_exp_dir: Path):
    experiments = scan_experiment_configs(mock_exp_dir, include_archived=False)

    # 2 runs for remove_rep_stopwords: one valid, one with NaN
    results_df = pl.DataFrame(
        {
            "source_file": [
                "fed_standard_aligned_umap-20260803-120000-123.csv",
                "fed_standard_aligned_umap-20260804-120000-456.csv",
            ],
            "experiment_id": [
                "fed_standard_aligned_umap",
                "fed_standard_aligned_umap",
            ],
            "dataset_label": ["fed", "fed"],
            "stopword_removal": [
                "remove_rep_stopwords",
                "remove_rep_stopwords",
            ],
            "u_mass": [-1.2, float("nan")],
            "c_v": [0.55, 0.40],
        }
    )

    matrix = build_coverage_matrix(experiments, results_df)
    aligned_row = matrix.filter(
        pl.col("experiment_name") == "fed_standard_aligned_umap"
    ).to_dicts()[0]

    assert aligned_row["remove_rep_stopwords"].startswith("⚠️ Partial Error")
    assert "1 valid" in aligned_row["remove_rep_stopwords"]
    assert "1 with NaNs" in aligned_row["remove_rep_stopwords"]
    assert aligned_row["coverage_score"] == "1/3"


def test_extract_model_name():
    assert extract_model_name("anes_standard_aligned_umap", "anes") == "aligned_umap"
    assert extract_model_name("fed_standard_baseline", "fed") == "baseline"
    assert extract_model_name("yelp_standard_mv_spectral", "yelp") == "mv_spectral"
    assert extract_model_name("trump_stemmed_mv_k_means", "trump") == "mv_k_means"
    # Autodetection without dataset label
    assert extract_model_name("gadarian_standard_stm") == "stm"
    assert extract_model_name("anes_standard_tritopic") == "tritopic"


def test_generate_slurm_command():
    # Local path default
    cmd1 = generate_slurm_command(
        dataset="anes",
        model="aligned_umap",
        condition="keep_rep_stopwords",
    )
    assert (
        cmd1
        == f"{DEFAULT_SLURM_SCRIPT_LOCAL} -d anes -m aligned_umap --keep-rep-stopwords"
    )

    # Cluster path
    cmd2 = generate_slurm_command(
        dataset="fed",
        model="baseline",
        condition="remove_rep_stopwords",
        script_path=DEFAULT_SLURM_SCRIPT_CLUSTER,
    )
    assert cmd2 == f"{DEFAULT_SLURM_SCRIPT_CLUSTER} -d fed -m baseline"

    # Stemmed condition
    cmd3 = generate_slurm_command(
        dataset=["anes", "fed"],
        model=["aligned_umap", "mv_spectral"],
        condition="stemmed",
        dry_run=True,
        auto_yes=True,
    )
    assert cmd3 == (
        f"{DEFAULT_SLURM_SCRIPT_LOCAL} -d anes,fed -m aligned_umap,mv_spectral"
        " --stemmed -n -y"
    )


def test_generate_grouped_slurm_commands(mock_exp_dir: Path):
    experiments = scan_experiment_configs(mock_exp_dir, include_archived=False)

    # Empty results -> all 3 conditions for both models are 'Not Run'
    results_df = pl.DataFrame()
    matrix = build_coverage_matrix(experiments, results_df)

    grouped = generate_grouped_slurm_commands(
        matrix,
        script_path=DEFAULT_SLURM_SCRIPT_CLUSTER,
        include_not_run=True,
    )

    assert "keep_rep_stopwords" in grouped
    assert "remove_rep_stopwords" in grouped
    assert "stemmed" in grouped

    # fed dataset has aligned_umap and baseline
    keep_cmds = grouped["keep_rep_stopwords"]
    assert len(keep_cmds) == 1
    assert "-d fed" in keep_cmds[0]
    assert "aligned_umap" in keep_cmds[0]
    assert "baseline" in keep_cmds[0]
    assert "--keep-rep-stopwords" in keep_cmds[0]
    assert keep_cmds[0].startswith(DEFAULT_SLURM_SCRIPT_CLUSTER)

    stemmed_cmds = grouped["stemmed"]
    assert len(stemmed_cmds) == 1
    assert "--stemmed" in stemmed_cmds[0]
