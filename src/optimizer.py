import polars as pl
from typing import Any, Dict, List
import logging
import itertools

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

    def _generate_hyperparameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generates all possible hyperparameter combinations from the model config.
        """
        hyperparameter_keys = []
        hyperparameter_values = []

        for key, value in self.model_config.items():
            if isinstance(value, list):
                hyperparameter_keys.append(key)
                hyperparameter_values.append(value)

        if not hyperparameter_keys:
            return [self.model_config]

        combinations = []
        for values in itertools.product(*hyperparameter_values):
            combination = self.model_config.copy()
            for i, key in enumerate(hyperparameter_keys):
                combination[key] = values[i]
            combinations.append(combination)
        
        return combinations


    def run(self) -> None:
        """
        Executes the optimization process: iterates through model configs,
        trains each one, and stores the evaluation metrics.
        """
        hyperparameter_combinations = self._generate_hyperparameter_combinations()
        num_combinations = len(hyperparameter_combinations)
        self.logger.info(f"Starting hyperparameter optimization for {num_combinations} models.")

        for i, model_config in enumerate(hyperparameter_combinations):
            model_id = model_config.get("name", f"model_{i+1}")
            self.logger.info(f"--- Training model [{model_id}] ({i+1}/{num_combinations}) ---")

            # 1. Create Model Instance
            topic_model = models.create_bertopic_instance(
                model_config=model_config,
                scaled_metadata=self.scaled_metadata,
                random_state=self.experiment_config["experiment"]["random_state"]
            )

            # 2. Train and Evaluate
            metrics, _ = training.train_and_evaluate(
                topic_model=topic_model,
                model_id=model_id,
                text=self.texts,
                embeddings=self.embeddings,
                config=self.experiment_config
            )
            
            # 3. Store results
            self.results.append(metrics)

        self.logger.info("Finished hyperparameter optimization.")

    def save_results(self, filepath: str) -> None:
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
