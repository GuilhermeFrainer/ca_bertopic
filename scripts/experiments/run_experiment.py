import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import datetime
import json
import logging
import traceback

import numpy as np
import polars as pl
from tqdm import tqdm

import src.data as data
import src.logger_config as logger_config
import src.make_table as make_table
import src.models as models
import src.run_provenance as run_provenance
import src.utils as utils
from src.document_assignments import AssignmentRun

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
TABLES_DIR = PROJECT_ROOT / "tables"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    parser = argparse.ArgumentParser(description="Run a specific ML experiment.")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Name of the experiment yaml file (e.g., experiment_2)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Override the sample size specified in the config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Run on a single specific random seed, overriding seeds in config.",
    )
    parser.add_argument(
        "--single-seed",
        action="store_true",
        help="Run only the first random seed from the config (ideal for dry runs).",
    )
    parser.add_argument(
        "--remove-rep-stopwords",
        action="store_true",
        default=True,
        help=(
            "Remove English stop words from BERTopic topic representations "
            "(c-TF-IDF) using CountVectorizer (default: True)."
        ),
    )
    parser.add_argument(
        "--keep-rep-stopwords",
        action="store_false",
        dest="remove_rep_stopwords",
        help="Keep English stop words in BERTopic topic representations.",
    )
    parser.add_argument(
        "--no-assignment-export",
        action="store_true",
        help="Disable document assignments; record disabled status.",
    )
    args = parser.parse_args()

    try:
        # Setup
        config = utils.load_config(args.exp, EXPERIMENTS_DIR)

        # Override sample size if requested
        if args.sample is not None:
            config["experiment"]["sample_size"] = args.sample
            config["experiment"]["name"] += f"_dry_run_{args.sample}"

        exp_name = config["experiment"]["name"]
        random_state = utils.get_random_state(config["experiment"]["random_state"])
        random_seeds = (
            random_state if isinstance(random_state, list) else [random_state]
        )
        if args.seed is not None:
            random_seeds = [args.seed]
        elif args.single_seed:
            random_seeds = [random_seeds[0]]

        primary_random_state = random_seeds[0]

        logger = logger_config.setup_logging(exp_name, LOG_DIR)
        if len(random_seeds) > 1:
            logger.info(
                f"Running experiment across {len(random_seeds)} seeds: {random_seeds}"
            )

        # Data loading using primary_random_state for consistent sampling
        logger.info("Loading and preparing data...")
        prepared = data.load_and_prep_data(
            config, random_state=primary_random_state, return_prepared=True
        )

        text, embeddings, scaled_metadata = prepared

        # Check for NaNs and warn if found
        if isinstance(scaled_metadata, pl.DataFrame):
            if scaled_metadata.width > 0:
                null_counts = scaled_metadata.null_count()
                nan_cols = [
                    col for col in scaled_metadata.columns if null_counts[col][0] > 0
                ]
                if nan_cols:
                    logger.warning(
                        f"Metadata contains null/NaN values in "
                        f"{len(nan_cols)} feature columns: {nan_cols}"
                    )
        elif scaled_metadata is not None and getattr(scaled_metadata, "size", 0) > 0:
            if np.isnan(scaled_metadata).any():
                nan_indices = np.where(np.isnan(scaled_metadata).any(axis=0))[0]
                logger.warning(
                    f"Metadata contains NaN values in "
                    f"{len(nan_indices)} feature columns."
                )
                logger.warning(f"NaN indices: {nan_indices.tolist()}")

        start_timestamp = datetime.datetime.now().isoformat()
        file_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset_name = Path(config["experiment"]["dataset_path"]).stem.replace(
            "_embeddings", ""
        )
        n_observations = len(text)

        # Model validation (using primary seed)
        logger.info("Validating model configurations...")
        models_config: list[dict] = config.get("models") or (
            [config["model"]] if "model" in config else []
        )

        for m_conf in models_config:
            m_id = m_conf.get("id", "Unknown")
            try:
                _ = models.create_topic_model_instance(
                    m_conf,
                    scaled_metadata,
                    primary_random_state,
                    remove_rep_stopwords=args.remove_rep_stopwords,
                )
            except Exception as e:
                logger.error(f"CRITICAL: Configuration error in model '{m_id}'")
                logger.error(f"Error details: {e}")
                raise e

        logger.info("All model configurations are valid. Starting training...")

        # Determine stopword status and output filenames ahead of training
        is_stemmed = "stemmed" in dataset_name.lower() or "stemmed" in exp_name.lower()
        if is_stemmed:
            stopword_status = "stemmed"
        elif args.remove_rep_stopwords:
            stopword_status = "remove_rep_stopwords"
        else:
            stopword_status = "keep_rep_stopwords"

        seeds_str = (
            "_".join(str(s) for s in random_seeds)
            if len(random_seeds) > 1
            else str(primary_random_state)
        )
        tag = stopword_status
        if tag in exp_name:
            fn_base = exp_name
        else:
            fn_base = f"{exp_name}_{tag}"
        results_filename = f"{fn_base}-{file_timestamp}-{seeds_str}"
        manifest_filename = f"{results_filename}_manifest.json"
        relative_manifest_path = f"logs/manifests/{manifest_filename}"

        baseline_config = next((m for m in models_config if m.get("is_baseline")), None)
        other_models = [m for m in models_config if m != baseline_config]

        results = []
        qualitative_dfs = []
        run_manifests = []

        for seed in random_seeds:
            logger.info(f"--- Running Seed: {seed} ---")
            baseline_n_topics = None

            # Run Baseline
            if baseline_config:
                b_id: str = baseline_config.get("id", "")
                logger.info(f"Running Baseline Model: {b_id} (seed {seed})")

                assignment_run = None
                try:
                    baseline_model = models.create_topic_model_instance(
                        baseline_config,
                        scaled_metadata,
                        seed,
                        remove_rep_stopwords=args.remove_rep_stopwords,
                    )

                    assignment_run = AssignmentRun(
                        OUTPUT_DIR / "document_assignments",
                        dataset_name,
                        prepared,
                        baseline_config,
                        {
                            "model_id": b_id,
                            "experiment_id": exp_name,
                            "dataset_name": dataset_name,
                            "seed": seed,
                            "stopword_removal": stopword_status,
                            "file_timestamp": file_timestamp,
                        },
                        config,
                        enabled=not args.no_assignment_export,
                    )
                    metrics, trained_model = assignment_run.execute(
                        topic_model=baseline_model,
                        model_id=b_id,
                        text=text,
                        embeddings=embeddings,
                        config=config,
                        scaled_metadata=scaled_metadata,
                    )

                    clustering_algo = baseline_config.get("clustering", {}).get(
                        "type", baseline_config.get("type", "baseline")
                    )
                    dim_red_config = baseline_config.get("dimensionality_reduction", {})
                    dim_red_algo = dim_red_config.get("type", "umap")

                    run_metadata = {
                        "experiment_id": exp_name,
                        "random_state": seed,
                        "clustering_algo": clustering_algo,
                        "dim_red_algo": dim_red_algo,
                        "n_observations": n_observations,
                        "timestamp": start_timestamp,
                        "file_timestamp": file_timestamp,
                        "dataset_name": dataset_name,
                        "stopword_removal": stopword_status,
                    }
                    run_metadata.update(assignment_run.links)
                    metrics.update(run_metadata)

                    # Collect Provenance
                    provenance = run_provenance.collect_run_provenance(
                        topic_model=trained_model,
                        model_config=baseline_config,
                        run_id=b_id,
                        run_manifest_path=relative_manifest_path,
                    )
                    metrics.update(provenance)
                    results.append(metrics)

                    run_manifests.append(
                        {
                            "run_id": b_id,
                            "model_name": b_id,
                            "seed": seed,
                            "status": "success",
                            **assignment_run.links,
                            "provenance": provenance,
                            "model_config": baseline_config,
                        }
                    )

                    qual_df = utils.extract_qualitative_data(
                        trained_model, b_id, run_metadata
                    )
                    qualitative_dfs.append(qual_df)

                    baseline_n_topics = metrics["n_topics"]
                    logger.info(
                        f"Baseline found {baseline_n_topics} topics for seed {seed}."
                    )
                except Exception:
                    tb_str = traceback.format_exc()
                    logger.error(
                        f"Failed during runtime of baseline {b_id} "
                        f"(seed {seed}):\n{tb_str}"
                    )
                    run_manifests.append(
                        {
                            "run_id": b_id,
                            "model_name": b_id,
                            "seed": seed,
                            "status": "failure",
                            **(assignment_run.links if assignment_run else {}),
                            "error": tb_str,
                            "model_config": baseline_config,
                        }
                    )

            # Run Remaining Models
            for model_config in tqdm(
                other_models, desc=f"Training models (seed {seed})"
            ):
                m_id = model_config.get("id", "")
                assignment_run = None
                try:
                    model_instance = models.create_topic_model_instance(
                        model_config,
                        scaled_metadata,
                        seed,
                        n_clusters=baseline_n_topics,
                        remove_rep_stopwords=args.remove_rep_stopwords,
                    )

                    assignment_run = AssignmentRun(
                        OUTPUT_DIR / "document_assignments",
                        dataset_name,
                        prepared,
                        model_config,
                        {
                            "model_id": m_id,
                            "experiment_id": exp_name,
                            "dataset_name": dataset_name,
                            "seed": seed,
                            "stopword_removal": stopword_status,
                            "file_timestamp": file_timestamp,
                        },
                        config,
                        enabled=not args.no_assignment_export,
                    )
                    metrics, trained_model = assignment_run.execute(
                        topic_model=model_instance,
                        model_id=m_id,
                        text=text,
                        embeddings=embeddings,
                        config=config,
                        scaled_metadata=scaled_metadata,
                    )

                    clustering_algo = model_config.get("clustering", {}).get(
                        "type", model_config.get("type", "tritopic")
                    )
                    dim_red_algo = model_config.get("dimensionality_reduction", {}).get(
                        "type", "tritopic_internal"
                    )

                    run_metadata = {
                        "experiment_id": exp_name,
                        "random_state": seed,
                        "clustering_algo": clustering_algo,
                        "dim_red_algo": dim_red_algo,
                        "n_observations": n_observations,
                        "timestamp": start_timestamp,
                        "file_timestamp": file_timestamp,
                        "dataset_name": dataset_name,
                        "stopword_removal": stopword_status,
                    }
                    run_metadata.update(assignment_run.links)
                    metrics.update(run_metadata)

                    # Collect Provenance
                    provenance = run_provenance.collect_run_provenance(
                        topic_model=trained_model,
                        model_config=model_config,
                        run_id=m_id,
                        run_manifest_path=relative_manifest_path,
                    )
                    metrics.update(provenance)
                    results.append(metrics)

                    run_manifests.append(
                        {
                            "run_id": m_id,
                            "model_name": m_id,
                            "seed": seed,
                            "status": "success",
                            **assignment_run.links,
                            "provenance": provenance,
                            "model_config": model_config,
                        }
                    )

                    qual_df = utils.extract_qualitative_data(
                        trained_model, m_id, run_metadata
                    )
                    qualitative_dfs.append(qual_df)

                except Exception:
                    tb_str = traceback.format_exc()
                    err_msg = f"Failed during runtime of {m_id} (seed {seed})."
                    if baseline_n_topics is not None and "ArpackError" in tb_str:
                        err_msg += (
                            " This is likely a numerical issue, possibly caused by "
                            f"forcing n_clusters={baseline_n_topics} on a model "
                            "that cannot support it with the given data."
                        )
                    logger.error(f"{err_msg}\n{tb_str}")
                    run_manifests.append(
                        {
                            "run_id": m_id,
                            "model_name": m_id,
                            "seed": seed,
                            "status": "failure",
                            **(assignment_run.links if assignment_run else {}),
                            "error": tb_str,
                            "model_config": model_config,
                        }
                    )
                    continue

        # Save Results
        results_df = pl.DataFrame(results)

        # Reorder columns for clean, consistent CSV schema
        all_cols = results_df.columns
        core_stats_cols = [
            "experiment_id",
            "random_state",
            "file_timestamp",
            "model_name",
            "dataset_name",
            "timestamp",
            "n_observations",
            "clustering_algo",
            "dim_red_algo",
            "duration_seconds",
            "n_topics",
            "outliers",
        ]
        exp_metrics = []
        if "experiment" in config:
            exp_metrics.extend(config["experiment"].get("coherence_metrics", []))
            exp_metrics.extend(config["experiment"].get("diversity_metrics", []))
        provenance_cols = run_provenance.PROVENANCE_COLUMNS
        param_cols = [
            col
            for col in all_cols
            if col not in core_stats_cols
            and col not in exp_metrics
            and col not in provenance_cols
        ]
        final_order = (
            [c for c in core_stats_cols if c in all_cols]
            + [p for p in param_cols if p in all_cols]
            + [m for m in exp_metrics if m in all_cols]
            + [pr for pr in provenance_cols if pr in all_cols]
        )
        final_order += [c for c in all_cols if c not in final_order]
        results_df = results_df.select(final_order)

        results_path = RESULTS_DIR / f"{results_filename}.csv"
        results_df.write_csv(results_path)

        logger.info(f"Experiment finished. Results at {results_path}")

        # Save Qualitative Data
        if qualitative_dfs:
            consolidated_qual_df = pl.concat(qualitative_dfs, how="diagonal")
            output_path = OUTPUT_DIR / f"{results_filename}.json"

            json_str = consolidated_qual_df.write_json()
            parsed_json = json.loads(json_str)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4)

            logger.info(f"Qualitative topic data saved at {output_path}")

        # Save Run Manifest
        manifest_dir = LOG_DIR / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / manifest_filename
        git_rev, git_dirty = run_provenance.get_git_info()
        manifest_payload = {
            "campaign_id": run_provenance.DEFAULT_CAMPAIGN_ID,
            "experiment_id": exp_name,
            "file_timestamp": file_timestamp,
            "code_revision": git_rev,
            "code_dirty": git_dirty,
            "dependency_lock_hash": run_provenance.get_dependency_lock_hash(),
            "runs": run_manifests,
        }
        run_provenance.save_run_manifest(manifest_path, manifest_payload)
        logger.info(f"Run manifest saved at {manifest_path}")

        latex_table = make_table.generate_latex_table(results_df)
        table_filename = f"{results_filename}.tex"
        table_path = TABLES_DIR / table_filename
        with open(table_path, "w") as f:
            f.write(latex_table)
        logger.info(f"Latex table saved at {table_path}")

    except Exception as e:
        logger = logging.getLogger("pipeline")
        logger.error(f"Pipeline crashed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
