import polars as pl
from typing import Any, Dict, List
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
        Generates hyperparameter combinations based on the user's heuristic:
        parameters to vary are in dimensionality_reduction.params or clustering.params.
        """
        import copy

        param_paths = []
        param_values = []

        def collect_params(component_name):
            component = self.model_config.get(component_name, {})
            params = component.get("params", {})
            for key, value in params.items():
                # A list of non-strings is considered a hyperparameter to vary
                if isinstance(value, list) and not all(isinstance(i, str) for i in value):
                    path = [component_name, "params", key]
                    param_paths.append(path)
                    param_values.append(value)

        collect_params("dimensionality_reduction")
        collect_params("clustering")

        if not param_paths:
            return [(self.model_config, {})]

        combinations = []
        for value_combination in itertools.product(*param_values):
            new_config = copy.deepcopy(self.model_config)
            varied_params = {}
            for path, value in zip(param_paths, value_combination):
                # Set nested item: e.g., new_config['clustering']['params']['n_clusters'] = value
                new_config[path[0]][path[1]][path[2]] = value
                varied_params[".".join(path)] = value
            combinations.append((new_config, varied_params))
        
        return combinations

    def run(self) -> None:
        """
        Executes the optimization process: iterates through model configs,
        trains each one, and stores the evaluation metrics.
        """
        hyperparameter_combinations = self._generate_hyperparameter_combinations()
        num_combinations = len(hyperparameter_combinations)
        self.logger.info(f"Starting hyperparameter optimization for {num_combinations} models.")

        for i, (model_config, varied_params) in enumerate(hyperparameter_combinations):
            model_id = self.model_config.get("id", "model")
            run_id = f"{model_id}_{i+1}"
            
            self.logger.info(f"--- Training model [{run_id}] ({i+1}/{num_combinations}) ---")
            self.logger.info(f"Varied Parameters: {varied_params}")

            # 1. Create Model Instance
            topic_model = models.create_bertopic_instance(
                model_config=model_config,
                scaled_metadata=self.scaled_metadata,
                random_state=self.experiment_config["experiment"]["random_state"]
            )

            # 2. Train and Evaluate
            metrics, _ = training.train_and_evaluate(
                topic_model=topic_model,
                model_id=run_id,
                text=self.texts,
                embeddings=self.embeddings,
                config=self.experiment_config
            )
            
            # 3. Store results, including the varied hyperparameters
            metrics.update(varied_params)
            self.results.append(metrics)

        self.logger.info("Finished hyperparameter optimization.")

    def save_results(self, filepath: str | pathlib.Path) -> None:
        """
        Saves the collected evaluation metrics to a CSV file.
        Args:
            filepath: The path to the output CSV file.
        """
        if not self.results:
            self.logger.warning("No results to save. Run the optimization first.")
            return

        df = pl.DataFrame(self.results)
        df.write_csv(filepath)
        self.logger.info(f"Results saved to {filepath}")
