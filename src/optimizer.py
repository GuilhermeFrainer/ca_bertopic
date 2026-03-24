import polars as pl
from typing import Any, Dict
import logging
import itertools
import pathlib

import src.models as models
import src.training as training


class Optimizer:
    """
    Orchestrates a hyperparameter search by training and evaluating multiple
    BERTopic models based on a single model architecture with multiple
    hyperparameter options.
    """

    def __init__(
        self,
        texts: list[str],
        embeddings: Any,
        scaled_metadata: Any,
        model_config: Dict[str, Any],
        experiment_config: Dict[str, Any]
    ):
        """
        Initializes the Optimizer.
        Args:
            texts: A list of strings with the texts to be analyzed
            embeddings: A np.ndarray with the document embeddings
            scaled_metadata: A np.ndarray with the document metadata
            model_config: A dictionary with the model architecture and 
                hyperparameters. Hyperparameters with multiple values 
                should be in a list.
            experiment_config: The global experiment configuration.
        """
        self.texts = texts
        self.embeddings = embeddings
        self.scaled_metadata = scaled_metadata
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.results = []
        self.logger = logging.getLogger("pipeline")

    def _generate_hyperparameter_combinations(self) -> list[tuple[dict, dict]]:
        """
        Generates all possible hyperparameter combinations for the search.
        """
        import copy

        param_paths, param_values = _collect_hyperparameters(self.model_config)

        if not param_paths:
            return [(self.model_config, {})]

        combinations = []
        # Create the Cartesian product of all hyperparameter values.
        # For each resulting combination, create a new model configuration.
        for value_combination in itertools.product(*param_values):
            new_config = copy.deepcopy(self.model_config)
            varied_params = {}
            for path, value in zip(param_paths, value_combination):
                # Set the specific hyperparameter value in the new config copy
                new_config[path[0]][path[1]][path[2]] = value
                # Keep track of the parameters that were varied for this run
                varied_params[".".join(path)] = value
            combinations.append((new_config, varied_params))
        
        return combinations

    def run(self) -> None:
        """
        Executes the optimization process: iterates through model configs,
        trains each one, and stores the evaluation metrics.
        """
        import datetime
        
        hyperparameter_combinations = self._generate_hyperparameter_combinations()
        num_combinations = len(hyperparameter_combinations)
        self.logger.info(f"Starting hyperparameter optimization for {num_combinations} models.")

        # New Metadata Capture
        start_timestamp = datetime.datetime.now().isoformat()
        dataset_name = pathlib.Path(self.experiment_config["experiment"]["dataset_path"]).stem
        n_observations = len(self.texts)

        for i, (model_config, varied_params) in enumerate(hyperparameter_combinations):
            model_id = self.model_config.get("id", "model")
            run_id = f"{model_id}_{i+1}"
            
            # Clean up param names for reporting
            cleaned_varied_params = {
                key.replace("clustering.params.", "").replace("dimensionality_reduction.params.", ""): value
                for key, value in varied_params.items()
            }
            self.logger.info(f"--- Training model [{run_id}] ({i+1}/{num_combinations}) ---")
            self.logger.info(f"Varied Parameters: {cleaned_varied_params}")

            # 1. Create Model Instance
            topic_model = models.create_bertopic_instance(
                model_config=model_config,
                scaled_metadata=self.scaled_metadata,
                random_state=self.experiment_config["experiment"]["random_state"]
            )

            # 2. Train and Evaluate
            try:
                metrics, _ = training.train_and_evaluate(
                    topic_model=topic_model,
                    model_id=run_id,
                    text=self.texts,
                    embeddings=self.embeddings,
                    config=self.experiment_config
                )
                
                # 3. Store results, including the varied hyperparameters and metadata
                metrics.update({
                    "clustering_algo": model_config["clustering"]["type"],
                    "dim_red_algo": model_config["dimensionality_reduction"]["type"],
                    "n_observations": n_observations,
                    "timestamp": start_timestamp,
                    "dataset_name": dataset_name,
                })
                metrics.update(cleaned_varied_params)
                self.results.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to train model [{run_id}] with params {cleaned_varied_params}: {e}")
                continue

        self.logger.info("Finished hyperparameter optimization.")

    def save_results(self, filepath: str | pathlib.Path, decimal_digits: int | None = None) -> None:
        """
        Saves the collected evaluation metrics to a CSV file.
        Args:
            filepath: The path to the output CSV file.
            decimal_digits: Optional number of digits for float precision in the CSV.
        """
        if not self.results:
            self.logger.warning("No results to save. Run the optimization first.")
            return

        df = pl.DataFrame(self.results)

        # Reorder columns based on user's desired output format
        all_cols = df.columns
        
        core_stats_cols = [
            "model_name", "dataset_name", "timestamp", "n_observations", 
            "clustering_algo", "dim_red_algo", "duration_seconds", "n_topics", "outliers"
        ]
        
        # Calculated metrics from the experiment config
        exp_metrics = []
        if "experiment" in self.experiment_config:
            exp_metrics.extend(self.experiment_config["experiment"].get("coherence_metrics", []))
            exp_metrics.extend(self.experiment_config["experiment"].get("diversity_metrics", []))

        # Varied parameter columns are what's left over
        param_cols = [
            col for col in all_cols if col not in core_stats_cols and col not in exp_metrics
        ]
        
        # New order: core stats, then params, then calculated metrics
        final_order = (
            [c for c in core_stats_cols if c in all_cols] +
            [p for p in param_cols if p in all_cols] +
            [m for m in exp_metrics if m in all_cols]
        )
        df = df.select(final_order)

        df.write_csv(filepath, float_precision=decimal_digits)
        self.logger.info(f"Results saved to {filepath}")


def _collect_hyperparameters(model_config: dict) -> tuple[list, list]:
    """
    Collects hyperparameter search spaces from the model configuration.

    This function identifies hyperparameters to be tuned by looking for lists
    of values or range specifications (dict with start, stop, step) within
    the 'dimensionality_reduction' and 'clustering' parameter sections of
    the model config.

    Args:
        model_config: The model configuration dictionary.

    Returns:
        A tuple containing two lists:
        - param_paths: A list of paths to the hyperparameters in the config dict.
        - param_values: A list of lists, where each inner list contains the
          values to be tested for the corresponding hyperparameter.
    """
    import numpy as np

    param_paths = []
    param_values = []

    def collect_from_component(component_name: str):
        component = model_config.get(component_name, {})
        params = component.get("params", {})
        for key, value in params.items():
            path = [component_name, "params", key]
            
            # A list of values is considered a hyperparameter to vary
            if isinstance(value, list) and len(value) > 1:
                param_paths.append(path)
                param_values.append(value)
            
            # A dictionary with start/stop is a range
            elif isinstance(value, dict) and "start" in value and "stop" in value:
                start = value["start"]
                stop = value["stop"]
                step = value.get("step", 1)
                
                # Use np.arange for float support and consistency
                generated_values = np.arange(start, stop, step).tolist()
                
                param_paths.append(path)
                param_values.append(generated_values)

    collect_from_component("dimensionality_reduction")
    collect_from_component("clustering")
    
    return param_paths, param_values