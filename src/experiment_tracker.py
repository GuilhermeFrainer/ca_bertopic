"""Experiment tracking and coverage analysis module.

This module provides functionality to scan experiment configuration files in
experiments/ and cross-reference them with result files in results/ to track
which experiments have been executed across representation/dataset conditions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl

CORE_EVAL_METRICS = [
    "u_mass",
    "c_v",
    "c_npmi",
    "c_uci",
    "irbo",
    "topic_diversity",
    "diversity",
]


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

            # Check for NaN or None in evaluation metric columns if present
            nan_metrics = []
            present_metrics = [m for m in CORE_EVAL_METRICS if m in r]
            for m in present_metrics:
                val = r.get(m)
                if val is None:
                    nan_metrics.append(m)
                elif isinstance(val, float) and math.isnan(val):
                    nan_metrics.append(m)
                elif isinstance(val, str) and val.strip().lower() in (
                    "nan",
                    "none",
                    "null",
                    "",
                ):
                    nan_metrics.append(m)

            has_nan = len(nan_metrics) > 0

            res_entry = {
                "source_file": src_file,
                "experiment_date": r.get("experiment_date"),
                "timestamp": r.get("timestamp"),
                "is_dry_run": is_dry_run,
                "sample_size": sample_size,
                "has_nan": has_nan,
                "nan_metrics": nan_metrics,
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
        has_condition_error = False

        for cond in conditions:
            res_list = matched_results.get((ds, c_name, cond), [])

            if not res_list:
                row[cond] = "❌ Not Run"
                details_map[cond] = {"status": "Not Run", "count": 0}
            else:
                full_runs = [r for r in res_list if not r["is_dry_run"]]
                dry_runs = [r for r in res_list if r["is_dry_run"]]

                if full_runs:
                    valid_runs = [r for r in full_runs if not r.get("has_nan", False)]
                    nan_runs = [r for r in full_runs if r.get("has_nan", False)]

                    if nan_runs and not valid_runs:
                        has_condition_error = True
                        row[cond] = f"❌ Error (NaNs in {len(nan_runs)} runs)"
                        details_map[cond] = {
                            "status": "Error",
                            "count": 0,
                            "error_count": len(nan_runs),
                            "dry_run_count": len(dry_runs),
                            "files": [r["source_file"] for r in res_list],
                            "nan_metrics": sorted(
                                list(
                                    {
                                        m
                                        for r in nan_runs
                                        for m in r.get("nan_metrics", [])
                                    }
                                )
                            ),
                        }
                    elif nan_runs and valid_runs:
                        completed_conds += 1
                        has_condition_error = True
                        row[cond] = (
                            f"⚠️ Partial Error ({len(valid_runs)} valid, "
                            f"{len(nan_runs)} with NaNs)"
                        )
                        details_map[cond] = {
                            "status": "Partial Error",
                            "count": len(valid_runs),
                            "error_count": len(nan_runs),
                            "dry_run_count": len(dry_runs),
                            "files": [r["source_file"] for r in res_list],
                            "nan_metrics": sorted(
                                list(
                                    {
                                        m
                                        for r in nan_runs
                                        for m in r.get("nan_metrics", [])
                                    }
                                )
                            ),
                        }
                    else:
                        completed_conds += 1
                        row[cond] = f"✅ Done ({len(full_runs)} runs)"
                        details_map[cond] = {
                            "status": "Completed",
                            "count": len(full_runs),
                            "dry_run_count": len(dry_runs),
                            "files": [r["source_file"] for r in res_list],
                        }
                else:
                    dry_valid = [r for r in dry_runs if not r.get("has_nan", False)]
                    dry_nan = [r for r in dry_runs if r.get("has_nan", False)]
                    if dry_nan and not dry_valid:
                        has_condition_error = True
                        row[cond] = f"❌ Error (NaNs in {len(dry_nan)} dry runs)"
                    elif dry_nan and dry_valid:
                        completed_conds += 1
                        has_condition_error = True
                        row[cond] = (
                            f"⚠️ Dry Run ({len(dry_valid)} valid, "
                            f"{len(dry_nan)} with NaNs)"
                        )
                    else:
                        completed_conds += 1
                        row[cond] = f"⚠️ Dry Run ({len(dry_runs)})"
                    details_map[cond] = {
                        "status": "Dry Run",
                        "count": 0,
                        "error_count": len(dry_nan),
                        "dry_run_count": len(dry_runs),
                        "files": [r["source_file"] for r in res_list],
                        "nan_metrics": sorted(
                            list({m for r in dry_nan for m in r.get("nan_metrics", [])})
                        ),
                    }

        row["completed_count"] = completed_conds
        row["coverage_score"] = f"{completed_conds}/3"
        if completed_conds == 3 and not has_condition_error:
            row["coverage_status"] = "Fully Completed"
        elif completed_conds == 3 and has_condition_error:
            row["coverage_status"] = "Completed with Errors"
        elif completed_conds > 0:
            row["coverage_status"] = "Partially Completed"
        elif has_condition_error:
            row["coverage_status"] = "Has Errors"
        else:
            row["coverage_status"] = "Not Run"

        row["run_details_json"] = json.dumps(details_map)
        matrix_data.append(row)

    if not matrix_data:
        return pl.DataFrame()

    return pl.DataFrame(matrix_data)


CONDITION_LABELS: dict[str, str] = {
    "keep_rep_stopwords": "Keep Stopwords",
    "remove_rep_stopwords": "Remove Stopwords",
    "stemmed": "Stemmed",
}

DEFAULT_SLURM_SCRIPT_LOCAL: str = "./scripts/pipelines/slurm/queue_exp.sh"
DEFAULT_SLURM_SCRIPT_CLUSTER: str = "./scripts/queue_exp.sh"


def extract_model_name(canonical_name: str, dataset_label: str = "") -> str:
    """Extracts the base model name from a canonical experiment identifier.

    Examples:
        anes_standard_aligned_umap -> aligned_umap
        fed_standard_baseline -> baseline
        yelp_standard_mv_spectral -> mv_spectral

    Args:
        canonical_name: Canonical experiment identifier.
        dataset_label: Optional dataset prefix if known.

    Returns:
        Clean model name matching queue_exp.sh models.
    """
    clean_name = canonical_name
    if dataset_label:
        prefixes = [
            f"{dataset_label}_standard_",
            f"{dataset_label}_stemmed_",
            f"{dataset_label}_",
        ]
        for p in prefixes:
            if clean_name.startswith(p):
                clean_name = clean_name[len(p) :]
                break
    else:
        for ds in ["anes", "fed", "gadarian", "yelp", "trump"]:
            prefixes = [f"{ds}_standard_", f"{ds}_stemmed_", f"{ds}_"]
            for p in prefixes:
                if clean_name.startswith(p):
                    clean_name = clean_name[len(p) :]
                    break

    # Also strip any leftover 'standard_' if present
    if clean_name.startswith("standard_"):
        clean_name = clean_name[len("standard_") :]

    return clean_name


def generate_slurm_command(
    dataset: str | list[str],
    model: str | list[str],
    condition: str,
    script_path: str = DEFAULT_SLURM_SCRIPT_LOCAL,
    dry_run: bool = False,
    auto_yes: bool = False,
    extra_flags: list[str] | None = None,
) -> str:
    """Generates a queue_exp.sh command to execute experiment(s) on SLURM.

    Args:
        dataset: Dataset label or list of dataset labels.
        model: Model identifier or list of model identifiers.
        condition: Condition key ('keep_rep_stopwords', 'remove_rep_stopwords',
            or 'stemmed').
        script_path: Script path (e.g. './scripts/pipelines/slurm/queue_exp.sh'
            or './scripts/queue_exp.sh').
        dry_run: If True, appends -n (--dry-run).
        auto_yes: If True, appends -y (--yes).
        extra_flags: Optional list of additional flags.

    Returns:
        Formatted SLURM queue command string.
    """
    if isinstance(dataset, str):
        ds_list = [d.strip() for d in dataset.split(",") if d.strip()]
    else:
        ds_list = list(dataset)

    if isinstance(model, str):
        m_list = [m.strip() for m in model.split(",") if m.strip()]
    else:
        m_list = list(model)

    ds_str = ",".join(dict.fromkeys(ds_list))
    m_str = ",".join(dict.fromkeys(m_list))

    parts = [script_path]
    if ds_str:
        parts.extend(["-d", ds_str])
    if m_str:
        parts.extend(["-m", m_str])

    if condition == "stemmed":
        parts.append("--stemmed")
    elif condition == "keep_rep_stopwords":
        parts.append("--keep-rep-stopwords")

    if dry_run:
        parts.append("-n")
    if auto_yes:
        parts.append("-y")
    if extra_flags:
        parts.extend(extra_flags)

    return " ".join(parts)


def generate_grouped_slurm_commands(
    coverage_df: pl.DataFrame,
    script_path: str = DEFAULT_SLURM_SCRIPT_LOCAL,
    include_not_run: bool = True,
    include_errors: bool = True,
    include_dry_runs: bool = False,
    target_conditions: list[str] | None = None,
    dry_run: bool = False,
    auto_yes: bool = False,
) -> dict[str, list[str]]:
    """Generates batched queue_exp.sh commands for missing/incomplete runs.

    Groups missing models per dataset for each condition to produce minimal, safe
    commands.

    Args:
        coverage_df: Polars DataFrame returned by build_coverage_matrix (or filtered
            slice).
        script_path: Path to queue_exp.sh.
        include_not_run: Include '❌ Not Run' conditions (default: True).
        include_errors: Include '❌ Error' and '⚠️ Partial Error' (default: True).
        include_dry_runs: Include '⚠️ Dry Run' (default: False).
        target_conditions: Conditions to inspect (default: all 3 conditions).
        dry_run: Append -n to generated commands.
        auto_yes: Append -y to generated commands.

    Returns:
        Dictionary mapping condition key to a list of executable queue_exp.sh
        command strings.
    """
    if coverage_df.is_empty():
        return {}

    if target_conditions is None:
        target_conditions = ["keep_rep_stopwords", "remove_rep_stopwords", "stemmed"]

    grouped_commands: dict[str, list[str]] = {cond: [] for cond in target_conditions}
    rows = coverage_df.to_dicts()

    for cond in target_conditions:
        # Map dataset -> list of missing model names (preserving order, no duplicates)
        ds_to_models: dict[str, list[str]] = {}

        for row in rows:
            if cond not in row:
                continue

            val = str(row.get(cond, ""))
            matched = False
            if val.startswith("❌ Not Run"):
                matched = include_not_run
            elif "Error" in val:
                matched = include_errors
            elif val.startswith("⚠️ Dry Run"):
                matched = include_dry_runs

            if matched:
                ds = str(row.get("dataset_label", "")).strip()
                canonical_name = str(row.get("experiment_name", "")).strip()
                model_name = extract_model_name(canonical_name, ds)

                if ds not in ds_to_models:
                    ds_to_models[ds] = []
                if model_name not in ds_to_models[ds]:
                    ds_to_models[ds].append(model_name)

        # Generate command for each dataset
        for ds, models in sorted(ds_to_models.items()):
            if models:
                cmd = generate_slurm_command(
                    dataset=ds,
                    model=models,
                    condition=cond,
                    script_path=script_path,
                    dry_run=dry_run,
                    auto_yes=auto_yes,
                )
                grouped_commands[cond].append(cmd)

    return grouped_commands
