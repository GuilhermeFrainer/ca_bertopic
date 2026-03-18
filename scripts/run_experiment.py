import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from tqdm import tqdm

import datetime
import argparse
import logging
import traceback

import src.utils as utils
import src.data as data
import src.training as training
import src.make_table as make_table
import src.models as models
import src.logger_config as logger_config


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
TABLES_DIR = PROJECT_ROOT / "tables"


def main():
    parser = argparse.ArgumentParser(description="Run a specific ML experiment.")
    parser.add_argument(
        "--exp", 
        type=str, 
        required=True, 
        help="Name of the experiment yaml file (e.g., experiment_2)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Override the sample size specified in the config file."
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

        logger = logger_config.setup_logging(exp_name, LOG_DIR)
        
        # Data loading
        # (We MUST do this first because models depend on scaled_metadata for init)
        logger.info("Loading and preparing data...")
        text, embeddings, scaled_metadata = data.load_and_prep_data(
            config, random_state=random_state)
        
        # New Metadata Capture
        start_timestamp = datetime.datetime.now().isoformat()
        dataset_name = Path(config["experiment"]["dataset_path"]).stem
        n_observations = len(text)

        # Model validation
        logger.info("Validating model configurations...")
        models_config: list[dict] = config["models"]
        
        for m_conf in models_config:
            m_id = m_conf.get("id", "Unknown")
            try:
                # Dry-run instantiation. 
                # We pass n_clusters=None just to ensure the parameters (strings/args) are valid.
                _ = models.create_bertopic_instance(m_conf, scaled_metadata, random_state)
            except Exception as e:
                logger.error(f"CRITICAL: Configuration error in model '{m_id}'")
                logger.error(f"Error details: {e}")
                # Crash immediately
                raise e 
        
        logger.info("All model configurations are valid. Starting training...")

        # We look for a model marked as baseline
        baseline_config = next(
            (m for m in models_config if m.get("is_baseline")), 
            None
        )
        
        # Filter out the baseline from the main list so we don't run it twice
        other_models = [m for m in models_config if m != baseline_config]
        
        results = []
        baseline_n_topics = None

        # Run Baseline
        if baseline_config:
            b_id: str = baseline_config.get('id', "")
            logger.info(f"Running Baseline Model: {b_id}")
            
            baseline_model = models.create_bertopic_instance(
                baseline_config, scaled_metadata, random_state
            )

            metrics, trained_model = training.train_and_evaluate(
                topic_model=baseline_model,
                model_id=b_id,
                text=text, 
                embeddings=embeddings, 
                config=config
            )

            # Add Metadata
            metrics.update({
                "clustering_algo": baseline_config["clustering"]["type"],
                "dim_red_algo": baseline_config["dimensionality_reduction"]["type"],
                "n_observations": n_observations,
                "timestamp": start_timestamp,
                "dataset_name": dataset_name,
            })
            results.append(metrics)
            
            baseline_n_topics = metrics["n_topics"]
            logger.info(f"Baseline found {baseline_n_topics} topics.")

        # Run Remaining Models
        for model_config in tqdm(other_models, desc="Training models"):
            m_id = model_config.get('id', "")
            try:
                model_instance = models.create_bertopic_instance(
                    model_config, 
                    scaled_metadata, 
                    random_state,
                    n_clusters=baseline_n_topics
                )

                metrics, _ = training.train_and_evaluate(
                    topic_model=model_instance,
                    model_id=m_id,
                    text=text, 
                    embeddings=embeddings, 
                    config=config
                )
                # Add Metadata
                metrics.update({
                    "clustering_algo": model_config["clustering"]["type"],
                    "dim_red_algo": model_config["dimensionality_reduction"]["type"],
                    "n_observations": n_observations,
                    "timestamp": start_timestamp,
                    "dataset_name": dataset_name,
                })
                results.append(metrics)

            except Exception as e:
                tb_str = traceback.format_exc()
                err_msg = f"Failed during runtime of {m_id}."
                # Check for the specific numerical error from the log
                if baseline_n_topics is not None and "ArpackError" in tb_str:
                    # Arpack error seems to be caused by making the model pick too many
                    # topics/clusters when there's not enough data to support it
                    #
                    # We log this information more explicitly to make debugging easier
                    err_msg += f" This is likely a numerical issue, possibly caused by forcing n_clusters={baseline_n_topics} on a model that cannot support it with the given data."
                
                logger.error(f"{err_msg}\n{tb_str}")
                continue


        # Save Results
        results_df = pl.DataFrame(results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_filename = f"{exp_name}-{timestamp}-{random_state}"
        results_path = RESULTS_DIR / f"{results_filename}.csv"
        results_df.write_csv(results_path)
        
        logger.info(f"Experiment finished. Results at {results_path}")

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


