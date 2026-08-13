"""Experiment tracking and coverage analysis module.

This module provides functionality to scan experiment configuration files in
experiments/ and cross-reference them with result files in results/ to track
which experiments have been executed across representation/dataset conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def scan_experiment_configs(
    exp_dir: str | Path = "experiments", include_archived: bool = False
) -> list[dict[str, Any]]:
    """Scans the experiments directory for YAML configuration files.

    Excludes dataset definition files (in experiments/datasets/) and optionally
    archived files (in experiments/archive/).

    Args:
        exp_dir: Path to the experiments directory.
        include_archived: Whether to include archived experiment YAMLs.

    Returns:
        A list of dictionaries representing discovered experiments.
    """
    exp_path = Path(exp_dir).resolve()
    if not exp_path.exists():
        return []

    yaml_files = list(exp_path.glob("**/*.yaml")) + list(exp_path.glob("**/*.yml"))
    discovered = []

    for file_path in yaml_files:
        rel_parts = file_path.relative_to(exp_path).parts

        # Skip base dataset parameter files
        if "datasets" in rel_parts:
            continue

        # Optionally skip archive
        if not include_archived and "archive" in rel_parts:
            continue

        stem = file_path.stem

        # Determine dataset folder / label
        if len(rel_parts) > 1:
            folder = rel_parts[0]
            if folder == "archive" and len(rel_parts) > 2:
                folder = rel_parts[1]

            if folder.endswith("_stemmed"):
                dataset_label = folder[:-8]
                is_stemmed_yaml = True
            else:
                dataset_label = folder
                is_stemmed_yaml = False
        else:
            # File in experiments/ root
            dataset_label = stem.split("_")[0]
            is_stemmed_yaml = "stemmed" in stem.lower()

        # Normalize to canonical experiment name
        canonical_name = stem
        if canonical_name.startswith(f"{dataset_label}_stemmed_"):
            canonical_name = canonical_name.replace(
                f"{dataset_label}_stemmed_", f"{dataset_label}_"
            )

        discovered.append(
            {
                "file_path": str(file_path),
                "stem": stem,
                "dataset_label": dataset_label,
                "canonical_name": canonical_name,
                "is_stemmed_yaml": is_stemmed_yaml,
                "is_archived": "archive" in rel_parts,
            }
        )

    return discovered


def classify_result_condition(
    source_file: str,
    exp_id: str = "",
    stopword_removal_col: str | None = None,
) -> str:
    """Classifies a result file/entry into one of the three dataset conditions.

    Conditions:
      - 'keep_rep_stopwords'
      - 'remove_rep_stopwords'
      - 'stemmed'

    Args:
        source_file: Filename of the result file.
        exp_id: Experiment ID string if available.
        stopword_removal_col: Explicit value from stopword_removal column if present.

    Returns:
        One of 'keep_rep_stopwords', 'remove_rep_stopwords', or 'stemmed'.
    """
    if stopword_removal_col and isinstance(stopword_removal_col, str):
        val = stopword_removal_col.lower()
        if val in ("keep_rep_stopwords", "none", "no_stopword_removal"):
            return "keep_rep_stopwords"
        if val in ("remove_rep_stopwords", "default"):
            return "remove_rep_stopwords"
        if "stemmed" in val:
            return "stemmed"

    fn_lower = source_file.lower()
    exp_lower = exp_id.lower()

    if "stemmed" in fn_lower or "stemmed" in exp_lower:
        return "stemmed"
    if "keep_rep_stopwords" in fn_lower or "no_stopword" in fn_lower:
        return "keep_rep_stopwords"

    return "remove_rep_stopwords"


def build_coverage_matrix(
    experiments: list[dict[str, Any]], results_df: pl.DataFrame
) -> pl.DataFrame:
    """Builds a coverage matrix cross-referencing experiments with results.

    Args:
        experiments: List of experiment dictionaries from scan_experiment_configs.
        results_df: Polars DataFrame of results loaded from load_all_results().

    Returns:
        Polars DataFrame containing experiment coverage status across conditions.
    """
    # 1. Group unique experiments by (dataset_label, canonical_name)
    exp_registry: dict[tuple[str, str], dict[str, Any]] = {}
    for exp in experiments:
        ds = exp["dataset_label"]
        c_name = exp["canonical_name"]
        key = (ds, c_name)

        if key not in exp_registry:
            exp_registry[key] = {
                "dataset_label": ds,
                "experiment_name": c_name,
                "yaml_standard": None,
                "yaml_stemmed": None,
                "is_archived": exp["is_archived"],
            }

        if exp["is_stemmed_yaml"]:
            exp_registry[key]["yaml_stemmed"] = exp["file_path"]
        else:
            exp_registry[key]["yaml_standard"] = exp["file_path"]

    # 2. Process results_df and map entries to (dataset, canonical_name, condition)
    # key -> list of result dicts
    matched_results: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    if not results_df.is_empty():
        rows = results_df.to_dicts()
        for r in rows:
            src_file = str(r.get("source_file") or "")
            exp_id_raw = r.get("experiment_id")
            exp_id = (
                str(exp_id_raw)
                if exp_id_raw is not None
                else (src_file.split("-")[0] if src_file else "")
            )

            ds_raw = r.get("dataset_label") or r.get("dataset_name")
            ds = (
                str(ds_raw)
                if ds_raw is not None
                else (src_file.split("_")[0] if src_file else "")
            )

            ds_clean = ds.replace("_embeddings", "").replace("_stemmed", "")
            stop_col = r.get("stopword_removal")
            cond = classify_result_condition(
                src_file, exp_id, stop_col if isinstance(stop_col, str) else None
            )

            # Strip dry_run suffix and tags for canonical matching
            exp_id_clean = exp_id.split("-")[0]
            for tag in ["_remove_rep_stopwords", "_keep_rep_stopwords"]:
                exp_id_clean = exp_id_clean.replace(tag, "")

            # Strip dry_run patterns if any
            parts = exp_id_clean.split("_dry_run_")
            exp_id_clean = parts[0]

            canonical_name = exp_id_clean.replace(
                f"{ds_clean}_stemmed_", f"{ds_clean}_"
            )

            is_dry_run = "_dry_run_" in src_file or "_dry_run_" in exp_id
            sample_size = None
            if is_dry_run and len(parts) > 1:
                try:
                    sample_size = int(parts[1].split("_")[0])
                except ValueError:
                    sample_size = None

            res_entry = {
                "source_file": src_file,
                "experiment_date": r.get("experiment_date"),
                "timestamp": r.get("timestamp"),
                "is_dry_run": is_dry_run,
                "sample_size": sample_size,
            }

            key = (ds_clean, canonical_name, cond)
            if key not in matched_results:
                matched_results[key] = []
            matched_results[key].append(res_entry)

    # 3. Construct final rows
    conditions = ["keep_rep_stopwords", "remove_rep_stopwords", "stemmed"]
    matrix_data = []

    for (ds, c_name), meta in sorted(exp_registry.items()):
        row: dict[str, Any] = {
            "dataset_label": ds,
            "experiment_name": c_name,
            "yaml_standard": meta["yaml_standard"],
            "yaml_stemmed": meta["yaml_stemmed"],
            "is_archived": meta["is_archived"],
        }

        completed_conds = 0
        details_map: dict[str, Any] = {}

        for cond in conditions:
            res_list = matched_results.get((ds, c_name, cond), [])

            if not res_list:
                row[cond] = "❌ Not Run"
                details_map[cond] = {"status": "Not Run", "count": 0}
            else:
                completed_conds += 1
                full_runs = [r for r in res_list if not r["is_dry_run"]]
                dry_runs = [r for r in res_list if r["is_dry_run"]]

                if full_runs:
                    row[cond] = f"✅ Done ({len(full_runs)} runs)"
                    details_map[cond] = {
                        "status": "Completed",
                        "count": len(full_runs),
                        "dry_run_count": len(dry_runs),
                        "files": [r["source_file"] for r in res_list],
                    }
                else:
                    row[cond] = f"⚠️ Dry Run ({len(dry_runs)})"
                    details_map[cond] = {
                        "status": "Dry Run",
                        "count": 0,
                        "dry_run_count": len(dry_runs),
                        "files": [r["source_file"] for r in res_list],
                    }

        row["completed_count"] = completed_conds
        row["coverage_score"] = f"{completed_conds}/3"
        if completed_conds == 3:
            row["coverage_status"] = "Fully Completed"
        elif completed_conds > 0:
            row["coverage_status"] = "Partially Completed"
        else:
            row["coverage_status"] = "Not Run"

        row["run_details_json"] = json.dumps(details_map)
        matrix_data.append(row)

    if not matrix_data:
        return pl.DataFrame()

    return pl.DataFrame(matrix_data)
