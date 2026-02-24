import polars as pl
from pathlib import Path
import datetime
import argparse
import traceback

import src.utils as utils
import src.data as data
import src.logger_config as logger_config
import src.make_table as make_table
from src.optimizer import Optimizer


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"
TABLES_DIR = BASE_DIR / "tables"


def main():
    parser = argparse.ArgumentParser(description="Run a hyperparameter optimization.")
    parser.add_argument(
        "--exp", 
        type=str, 
        required=True, 
        help="Name of the optimization yaml file (e.g., yelp_opt_spectral)"
    )
    args = parser.parse_args()

    logger = None  # Initialize logger to None

    try:
        # Setup
        config = utils.load_config(args.exp, EXPERIMENTS_DIR)
        exp_name = config["experiment"]["name"]
        random_state = utils.get_random_state(config["experiment"]["random_state"])

        logger = logger_config.setup_logging(exp_name, LOG_DIR)
        
        # Data loading
        logger.info("Loading and preparing data...")
        text, embeddings, scaled_metadata = data.load_and_prep_data(
            config, random_state=random_state)

        # Model configuration
        logger.info("Loading model configuration for optimization...")
        model_config = config.get("model")
        if not model_config:
            raise ValueError("Configuration file must contain a 'model' section for optimization.")

        # Initialize and run optimizer
        optimizer = Optimizer(
            texts=text,
            embeddings=embeddings,
            scaled_metadata=scaled_metadata,
            model_config=model_config,
            experiment_config=config
        )
        optimizer.run()

        # Save Results
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_filename = f"{exp_name}-{timestamp}-{random_state}"
        results_path = RESULTS_DIR / f"{results_filename}.csv"
        optimizer.save_results(results_path)
        
        logger.info(f"Optimization finished. Results at {results_path}")

        # Generate and save LaTeX table if there are results
        if optimizer.results:
            results_df = pl.DataFrame(optimizer.results)
            latex_table = make_table.generate_latex_table(results_df)
            table_filename = f"{results_filename}.tex"
            table_path = TABLES_DIR / table_filename
            with open(table_path, "w") as f:
                f.write(latex_table)
            logger.info(f"Latex table saved at {table_path}")
        else:
            logger.warning("No results were generated, skipping LaTeX table creation.")

    except Exception as e:
        if logger:
            logger.error(f"Pipeline crashed: {e}", exc_info=True)
        else:
            # If logger setup fails, print to stderr
            print(f"Pipeline crashed before logger was configured: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
