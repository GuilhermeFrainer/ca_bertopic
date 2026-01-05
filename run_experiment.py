import polars as pl
from tqdm import tqdm

from pathlib import Path
import datetime
import argparse
import logging

import src.utils as utils
import src.data as data
import src.training as training


EXPERIMENTS_DIR = Path("./experiments")
OUTPUT_DIR = Path("./output")
LOG_DIR = Path("./logs")


def main():
    parser = argparse.ArgumentParser(description="Run a specific ML experiment.")
    parser.add_argument(
        "--exp", 
        type=str, 
        required=True, 
        help="Name of the experiment yaml file (e.g., experiment_2)"
    )
    args = parser.parse_args()

    try:
        # 1. Setup Config & Logs
        config = utils.load_config(args.exp, EXPERIMENTS_DIR)
        exp_name = config["experiment"]["name"]
        logger = utils.setup_logging(exp_name, LOG_DIR)
        
        logger.info("Loading and preparing data...")
        text, embeddings, scaled_metadata = data.load_and_prep_data(config)

        # 2. Separate Baseline from Others
        models_config = config["models"]
        
        # We look for a model marked as baseline OR strictly named "vanilla"
        baseline_config = next(
            (m for m in models_config if m.get("is_baseline") or m.get("id") == "vanilla"), 
            None
        )
        
        # Filter out the baseline from the main list so we don't run it twice
        other_models = [m for m in models_config if m != baseline_config]
        
        results = []
        baseline_n_topics = None

        # 3. Run Baseline First (if it exists)
        if baseline_config:
            logger.info(f"Running Baseline Model: {baseline_config.get('id')}")
            metrics, trained_model = training.train_and_evaluate(
                baseline_config, text, embeddings, scaled_metadata, config
            )
            results.append(metrics)
            
            # Extract n_topics
            baseline_n_topics = len(trained_model.get_topic_info()) - 1
            logger.info(f"Baseline found {baseline_n_topics} topics.")

        # 4. Run Remaining Models
        for model_config in tqdm(other_models, desc="Training models"):
            try:
                metrics, _ = training.train_and_evaluate(
                    model_config, 
                    text, 
                    embeddings, 
                    scaled_metadata, 
                    config,
                    baseline_topics=baseline_n_topics
                )
                results.append(metrics)
            except Exception as e:
                logger.error(f"Failed model {model_config.get('id')}: {e}")
                # Don't raise if you want other models to keep running
                continue

        # 5. Save Results
        results_df = pl.DataFrame(results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = OUTPUT_DIR / f"{exp_name}-{timestamp}.csv"
        results_df.write_csv(results_path)
        
        logger.info(f"Experiment finished. Results at {results_path}")

    except Exception as e:
        logger = logging.getLogger("pipeline")
        logger.error(f"Pipeline crashed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

