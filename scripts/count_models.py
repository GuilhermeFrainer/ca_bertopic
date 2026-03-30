import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils as utils
from src.optimizer import generate_hyperparameter_combinations, clean_varied_params

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

def main():
    parser = argparse.ArgumentParser(description="Count and list model configurations in an experiment.")
    parser.add_argument(
        "--exp", 
        type=str, 
        required=True, 
        help="Name of the optimization yaml file (e.g., yelp_opt_spectral)"
    )
    args = parser.parse_args()

    try:
        config = utils.load_config(args.exp, EXPERIMENTS_DIR)
        model_config = config.get("model")
        if not model_config:
            print(f"Error: Configuration file '{args.exp}' must contain a 'model' section for optimization.")
            return

        combinations = generate_hyperparameter_combinations(model_config)
        print(f"Total model configurations for '{args.exp}': {len(combinations)}")
        print("-" * 50)

        for i, (_, varied_params) in enumerate(combinations, 1):
            cleaned_params = clean_varied_params(varied_params)
            param_str = ", ".join([f"{k}: {v}" for k, v in cleaned_params.items()])
            print(f"Model {i:3}: {param_str if param_str else 'Default parameters'}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
