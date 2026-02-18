import polars as pl
from tqdm import tqdm

from pathlib import Path
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


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"
TABLES_DIR = BASE_DIR / "tables"


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
        # 1. Setup
        config = utils.load_config(args.exp, EXPERIMENTS_DIR)
        exp_name = config["experiment"]["name"]
        random_state = utils.get_random_state(config["experiment"]["random_state"])

        logger = logger_config.setup_logging(exp_name, LOG_DIR)
        
        # 2. Load Data 
        # (We MUST do this first because models depend on scaled_metadata for init)
        logger.info("Loading and preparing data...")
        text, embeddings, scaled_metadata = data.load_and_prep_data(
            config, random_state=random_state)

        # ---------------------------------------------------------
        # 3. FAIL FAST: Validation Loop
        # ---------------------------------------------------------
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

        # We look for a model marked as baseline OR strictly named "vanilla"
        baseline_config = next(
            (m for m in models_config if m.get("is_baseline")), 
            None
        )
        
        # Filter out the baseline from the main list so we don't run it twice
        other_models = [m for m in models_config if m != baseline_config]
        
        results = []
        baseline_n_topics = None

        # 5. Run Baseline
        if baseline_config:
            b_id: str = baseline_config.get('id', "")
            logger.info(f"Running Baseline Model: {b_id}")
            
            # Real Instantiation for Baseline
            baseline_model = models.create_bertopic_instance(
                baseline_config, scaled_metadata, random_state
            )

            metrics, trained_model = training.train_and_evaluate(
                topic_model=baseline_model,  # Pass object
                model_id=b_id,
                text=text, 
                embeddings=embeddings, 
                config=config
            )
            results.append(metrics)
            
            baseline_n_topics = metrics["n_topics"]
            logger.info(f"Baseline found {baseline_n_topics} topics.")

        # 6. Run Remaining Models
        for model_config in tqdm(other_models, desc="Training models"):
            m_id = model_config.get('id', "")
            try:
                # Real Instantiation for Others (using baseline topic count)
                model_instance = models.create_bertopic_instance(
                    model_config, 
                    scaled_metadata, 
                    random_state,
                    n_clusters=baseline_n_topics # Injecting the dependency here
                )

                metrics, _ = training.train_and_evaluate(
                    topic_model=model_instance, # Pass object
                    model_id=m_id,
                    text=text, 
                    embeddings=embeddings, 
                    config=config
                )
                results.append(metrics)

            except Exception as e:
                logger.error(f"Failed during runtime of {m_id}\n {traceback.format_exc()}")
                continue


        # 5. Save Results
        results_df = pl.DataFrame(results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_filename = f"{exp_name}-{timestamp}-{random_state}"
        results_path = RESULTS_DIR / f"{results_filename}.csv"
        results_df.write_csv(results_path)
        
        logger.info(f"Experiment finished. Results at {results_path}")

        latex_table = make_table.generate_latex_table(results_df)
        #latex_table = results_df.to_pandas().to_latex()
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

