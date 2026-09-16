from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

import src.make_table as make_table
import src.run_provenance as run_provenance
from src.optimizer import Optimizer


def test_get_git_info():
    commit, dirty = run_provenance.get_git_info()
    assert isinstance(commit, str)
    assert len(commit) > 0
    assert isinstance(dirty, bool)


def test_get_dependency_lock_hash():
    lock_hash = run_provenance.get_dependency_lock_hash()
    assert isinstance(lock_hash, str)
    assert len(lock_hash) == 16 or lock_hash == "unknown"


def test_compute_config_hash():
    cfg1 = {"a": 1, "b": [1, 2, 3]}
    cfg2 = {"b": [1, 2, 3], "a": 1}
    cfg3 = {"a": 2, "b": [1, 2, 3]}

    h1 = run_provenance.compute_config_hash(cfg1)
    h2 = run_provenance.compute_config_hash(cfg2)
    h3 = run_provenance.compute_config_hash(cfg3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_observe_reducer_output_dim():
    # None checks
    assert run_provenance.observe_reducer_output_dim(None) is None
    mock_empty = MagicMock()
    mock_empty.umap_model = None
    assert run_provenance.observe_reducer_output_dim(mock_empty) is None

    # Plain UMAP / AppendUMAP with embedding_
    mock_umap = MagicMock()
    mock_umap.umap_model.embedding_ = np.zeros((20, 5))
    assert run_provenance.observe_reducer_output_dim(mock_umap) == 5

    # AlignedUMAPWrapper with training_embeddings
    mock_aligned = MagicMock()
    del mock_aligned.umap_model.embedding_
    mock_aligned.umap_model.training_embeddings = np.zeros((20, 5))
    assert run_provenance.observe_reducer_output_dim(mock_aligned) == 5

    # PCA with n_components_
    mock_pca = MagicMock()
    del mock_pca.umap_model.embedding_
    del mock_pca.umap_model.training_embeddings
    mock_pca.umap_model.n_components_ = 5
    assert run_provenance.observe_reducer_output_dim(mock_pca) == 5

    # PCA with components_
    mock_pca2 = MagicMock()
    del mock_pca2.umap_model.embedding_
    del mock_pca2.umap_model.training_embeddings
    del mock_pca2.umap_model.n_components_
    mock_pca2.umap_model.components_ = np.zeros((5, 100))
    assert run_provenance.observe_reducer_output_dim(mock_pca2) == 5


def test_collect_run_provenance_schema():
    model_config = {
        "type": "bertopic",
        "dimensionality_reduction": {
            "type": "umap",
            "params": {
                "n_components": 5,
                "n_neighbors": 15,
                "metric": "cosine",
                "min_dist": 0.0,
                "low_memory": False,
            },
        },
        "clustering": {
            "type": "hdbscan",
            "params": {
                "min_cluster_size": 10,
                "min_samples": 5,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "prediction_data": True,
            },
        },
    }

    mock_model = MagicMock()
    mock_model.umap_model.n_components = 5
    mock_model.umap_model.n_neighbors = 15
    mock_model.umap_model.metric = "cosine"
    mock_model.umap_model.min_dist = 0.0
    mock_model.umap_model.low_memory = False
    mock_model.umap_model.embedding_ = np.zeros((50, 5))
    del mock_model.hdbscan_model.model
    mock_model.hdbscan_model.min_cluster_size = 10
    mock_model.hdbscan_model.min_samples = 5
    mock_model.hdbscan_model.metric = "euclidean"
    mock_model.hdbscan_model.cluster_selection_method = "eom"
    mock_model.hdbscan_model.prediction_data = True

    prov = run_provenance.collect_run_provenance(
        topic_model=mock_model,
        model_config=model_config,
        run_id="test_model_1",
        run_manifest_path="logs/manifests/test_manifest.json",
    )

    # Verify all columns in PROVENANCE_COLUMNS are present
    for col in run_provenance.PROVENANCE_COLUMNS:
        assert col in prov, f"Missing provenance column: {col}"

    assert prov["result_schema_version"] == 2
    assert prov["campaign_id"] == "bertopic_defaults_v2"
    assert prov["run_status"] == "success"
    assert prov["dim_red_n_components"] == 5
    assert prov["dim_red_output_dim"] == 5
    assert prov["dim_red_n_neighbors"] == 15
    assert prov["dim_red_metric"] == "cosine"
    assert prov["dim_red_min_dist"] == 0.0
    assert prov["dim_red_low_memory"] is False
    assert prov["cluster_min_cluster_size"] == 10
    assert prov["cluster_min_samples"] == 5
    assert prov["cluster_metric"] == "euclidean"
    assert prov["cluster_selection_method"] == "eom"
    assert prov["cluster_prediction_data"] is True
    assert prov["normalize_text_view"] is False
    assert prov["run_manifest_path"] == "logs/manifests/test_manifest.json"


def test_save_run_manifest(tmp_path):
    manifest_path = tmp_path / "test_manifest.json"
    data = {
        "campaign_id": "bertopic_defaults_v2",
        "runs": [{"run_id": "m1", "status": "success"}],
    }
    saved_path = run_provenance.save_run_manifest(manifest_path, data)
    assert saved_path.exists()

    import json

    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["campaign_id"] == "bertopic_defaults_v2"
    assert len(loaded["runs"]) == 1


def test_full_precision_csv_and_table_display(tmp_path):
    # Full precision floats
    high_precision_val_pos = 0.1234567890123456
    high_precision_val_neg = -12.34567890123456

    csv_path = tmp_path / "test_precision.csv"

    # Optimizer mock results with provenance and metrics
    optimizer = Optimizer(
        texts=["text1", "text2"],
        embeddings=np.zeros((2, 10)),
        scaled_metadata=None,
        model_config={"type": "bertopic"},
        experiment_config={
            "experiment": {
                "coherence_metrics": ["c_v"],
                "diversity_metrics": ["irbo"],
            }
        },
        experiment_id="test_exp",
        random_state=42,
        file_timestamp="20260916-120000",
    )

    optimizer.results = [
        {
            "experiment_id": "test_exp",
            "random_state": 42,
            "file_timestamp": "20260916-120000",
            "model_name": "test_model",
            "dataset_name": "test_ds",
            "timestamp": "2026-09-16T12:00:00",
            "n_observations": 2,
            "clustering_algo": "hdbscan",
            "dim_red_algo": "umap",
            "duration_seconds": 1.234,
            "n_topics": 5,
            "outliers": 0,
            "c_v": high_precision_val_pos,
            "irbo": high_precision_val_neg,
            "result_schema_version": 2,
            "campaign_id": "bertopic_defaults_v2",
            "run_status": "success",
            "dim_red_n_components": 5,
            "dim_red_output_dim": 5,
            "dim_red_n_neighbors": 15,
            "dim_red_metric": "cosine",
            "dim_red_min_dist": 0.0,
            "dim_red_low_memory": False,
            "cluster_min_cluster_size": 10,
            "cluster_min_samples": 5,
            "cluster_metric": "euclidean",
            "cluster_selection_method": "eom",
            "cluster_prediction_data": True,
            "normalize_text_view": False,
            "resolved_config_hash": "abcd1234efgh5678",
            "run_manifest_path": "logs/manifests/test_manifest.json",
            "code_revision": "abc1234",
            "code_dirty": False,
            "dependency_lock_hash": "12345678abcdef00",
        }
    ]

    optimizer.save_results(csv_path, decimal_digits=3)

    # 1. Verify CSV contains full precision without truncation
    loaded_df = pl.read_csv(csv_path, infer_schema_length=None)
    assert loaded_df["c_v"][0] == pytest.approx(high_precision_val_pos, abs=1e-15)
    assert loaded_df["irbo"][0] == pytest.approx(high_precision_val_neg, abs=1e-15)

    raw_csv_text = csv_path.read_text(encoding="utf-8")
    assert "0.1234567890123456" in raw_csv_text
    assert "-12.34567890123456" in raw_csv_text

    # 2. Verify LaTeX table formats at 3 decimals (unchanged display precision)
    latex = make_table.generate_latex_table(loaded_df)
    assert "0.123" in latex
    assert "-12.346" in latex
    # Provenance cols must NOT appear as table columns in LaTeX
    assert "result_schema_version" not in latex
    assert "campaign_id" not in latex

    # 3. Verify Great Tables excludes provenance columns from metric calculations
    gt = make_table.generate_gt_table(loaded_df)
    # The GT internal table data should have metric_cols only containing c_v and irbo
    # (dim_red_output_dim, result_schema_version, etc. are excluded)
    gt_df = gt._tbl_data
    for col in run_provenance.PROVENANCE_COLUMNS:
        assert col not in gt_df.columns
    gt_html = gt.as_raw_html()
    for col in run_provenance.PROVENANCE_COLUMNS:
        assert col not in gt_html


def test_campaign_isolation_on_merge(tmp_path):
    # If an existing results file belongs to an old campaign or has no campaign_id,
    # save_results should NOT append to it; it should save to a separate file.
    legacy_csv = tmp_path / "test_legacy.csv"
    legacy_df = pl.DataFrame(
        {
            "experiment_id": ["legacy_exp"],
            "model_name": ["legacy_model"],
            "n_topics": [10],
            "c_v": [0.55],
        }
    )
    legacy_df.write_csv(legacy_csv)

    optimizer = Optimizer(
        texts=["text1", "text2"],
        embeddings=np.zeros((2, 10)),
        scaled_metadata=None,
        model_config={"type": "bertopic"},
        experiment_config={"experiment": {}},
        experiment_id="test_exp",
        random_state=42,
        file_timestamp="20260916-120000",
    )
    optimizer.results = [
        {
            "experiment_id": "test_exp",
            "model_name": "corrected_model",
            "campaign_id": "bertopic_defaults_v2",
            "result_schema_version": 2,
            "c_v": 0.65,
        }
    ]

    optimizer.save_results(legacy_csv)

    # The legacy file must NOT be modified
    legacy_after = pl.read_csv(legacy_csv)
    assert len(legacy_after) == 1
    assert legacy_after["model_name"][0] == "legacy_model"

    # The corrected results must be in a separate file with campaign or v2 suffix
    separated_csv = tmp_path / "test_legacy_bertopic_defaults_v2.csv"
    assert separated_csv.exists()
    separated_df = pl.read_csv(separated_csv)
    assert len(separated_df) == 1
    assert separated_df["model_name"][0] == "corrected_model"


def test_optimizer_provenance_integration(tmp_path):
    """
    Tests that Optimizer.run() end-to-end captures fitted dimensionality and provenance,
    and save_results creates the full precision CSV and JSON run manifest.
    """
    model_config = {
        "id": "test_prov_model",
        "dimensionality_reduction": {
            "type": "umap",
            "params": {
                "n_components": 5,
                "n_neighbors": 3,
                "min_dist": 0.0,
                "metric": "cosine",
            },
        },
        "clustering": {
            "type": "hdbscan",
            "params": {"min_cluster_size": 2},
        },
        "bertopic": {
            "params": {"nr_topics": 2},
        },
    }

    experiment_config = {
        "experiment": {
            "dataset_path": "data/mock_dataset.parquet",
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }

    texts = [
        "apple banana orange fruit salad healthy sweet",
        "pear apple grape fruit smoothie fresh organic",
        "car truck vehicle engine motor automotive diesel",
        "bus train vehicle transit motor commute transportation",
        "python code software programming test developer algorithm",
        "java script software developer coding backend engineering",
        "biology cells organisms genetics molecular DNA science",
        "chemistry atoms molecules reaction elements laboratory test",
        "physics gravity quantum relativity mechanics theoretical law",
        "finance stocks investment banking capital economy market",
        "history ancient roman empire civilization archaeology war",
        "music guitar drums concert melody rhythm acoustic",
        "painting canvas watercolor artist gallery exhibition portrait",
        "cooking recipe kitchen baking ingredients spices dining",
        "astronomy planets galaxies telescope cosmos universe solar",
    ]
    embeddings = np.random.rand(len(texts), 10)
    scaled_metadata = np.random.rand(len(texts), 4)

    optimizer = Optimizer(
        texts=texts,
        embeddings=embeddings,
        scaled_metadata=scaled_metadata,
        model_config=model_config,
        experiment_config=experiment_config,
        experiment_id="test_prov_opt",
        random_state=42,
        file_timestamp="20260916-150000",
    )

    optimizer.run()

    assert len(optimizer.results) == 1
    res = optimizer.results[0]
    assert res["result_schema_version"] == 2
    assert res["campaign_id"] == "bertopic_defaults_v2"
    assert res["dim_red_output_dim"] == 5
    assert res["dim_red_n_components"] == 5
    assert res["dim_red_n_neighbors"] == 3
    assert res["dim_red_metric"] == "cosine"
    assert res["cluster_min_cluster_size"] == 2
    assert res["run_status"] == "success"

    # Test saving results and manifest
    results_csv = tmp_path / "results" / "test_prov_opt.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    optimizer.save_results(results_csv)
    assert results_csv.exists()

    saved_df = pl.read_csv(results_csv, infer_schema_length=None)
    assert "dim_red_output_dim" in saved_df.columns
    assert saved_df["dim_red_output_dim"][0] == 5
    assert saved_df["campaign_id"][0] == "bertopic_defaults_v2"
    assert (
        saved_df["run_manifest_path"][0] == "logs/manifests/test_prov_opt_manifest.json"
    )

    manifest_json = tmp_path / "logs" / "manifests" / "test_prov_opt_manifest.json"
    assert manifest_json.exists()

    import json

    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["campaign_id"] == "bertopic_defaults_v2"
    assert len(manifest_data["runs"]) == 1
    assert manifest_data["runs"][0]["provenance"]["dim_red_output_dim"] == 5
    assert manifest_data["runs"][0]["model_name"] == "test_prov_model_1"

    # Bidirectional traceability verification:
    # 1. CSV row -> Manifest entry
    row = saved_df.to_dicts()[0]
    matched_run = next(
        r
        for r in manifest_data["runs"]
        if r["model_name"] == row["model_name"]
        and r["seed"] == int(row["random_state"])
    )
    assert (
        matched_run["provenance"]["resolved_config_hash"] == row["resolved_config_hash"]
    )

    # 2. Manifest entry -> CSV row
    run_entry = manifest_data["runs"][0]
    filtered_df = saved_df.filter(
        (pl.col("model_name") == run_entry["model_name"])
        & (pl.col("random_state") == run_entry["seed"])
    )
    assert len(filtered_df) == 1
    assert filtered_df["run_manifest_path"][0] == row["run_manifest_path"]


def test_get_git_info_env_override(monkeypatch):
    """Verify that get_git_info prioritizes GIT_COMMIT_REV and GIT_DIRTY env vars."""
    monkeypatch.setenv("GIT_COMMIT_REV", "cluster_commit_12345")
    monkeypatch.setenv("GIT_DIRTY", "true")
    rev, dirty = run_provenance.get_git_info()
    assert rev == "cluster_commit_12345"
    assert dirty is True

    monkeypatch.setenv("GIT_DIRTY", "false")
    rev, dirty = run_provenance.get_git_info()
    assert rev == "cluster_commit_12345"
    assert dirty is False
