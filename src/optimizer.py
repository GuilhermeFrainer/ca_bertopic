import copy
import itertools
import json
import logging
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl

import src.models as models
import src.training as training


def collect_hyperparameters(
    model_config: Dict[str, Any],
) -> Tuple[List[List[str]], List[List[Any]]]:
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
    param_paths = []
    param_values = []

    def collect_from_component(component_name: str):
        component = model_config.get(component_name, {})
        params = component.get("params", {})
        # Sort keys to ensure deterministic order of combinations
        for key in sorted(params.keys()):
            value = params[key]
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
    collect_from_component("bertopic")

    return param_paths, param_values


def generate_hyperparameter_combinations(
    model_config: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Generates all possible hyperparameter combinations for the search.
    """
    param_paths, param_values = collect_hyperparameters(model_config)

    if not param_paths:
        return [(model_config, {})]

    combinations = []
    # Create the Cartesian product of all hyperparameter values.
    # For each resulting combination, create a new model configuration.
    for value_combination in itertools.product(*param_values):
        new_config = copy.deepcopy(model_config)
        varied_params = {}
        for path, value in zip(param_paths, value_combination):
            # Set the specific hyperparameter value in the new config copy
            new_config[path[0]][path[1]][path[2]] = value
            # Keep track of the parameters that were varied for this run
            varied_params[".".join(path)] = value
        combinations.append((new_config, varied_params))

    return combinations


def clean_varied_params(varied_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans up parameter names for reporting and results.
    """
    return {
        key.replace("clustering.params.", "")
        .replace("dimensionality_reduction.params.", "")
        .replace("bertopic.params.", ""): value
        for key, value in varied_params.items()
    }


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
        experiment_config: Dict[str, Any],
        experiment_id: str,
        random_state: int,
        file_timestamp: str,
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
            experiment_id: The identifier for the experiment.
            random_state: The random seed used for the experiment.
            file_timestamp: The timestamp used in the results filename.
        """
        self.texts = texts
        self.embeddings = embeddings
        self.scaled_metadata = scaled_metadata
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.experiment_id = experiment_id
        self.random_state = random_state
        self.file_timestamp = file_timestamp
        self.results = []
        self.qualitative_results = []
        self.logger = logging.getLogger("pipeline")

    def run(self, start_index: int = 0, target_index: int | None = None) -> None:
        """
        Executes the optimization process: iterates through model configs,
        trains each one, and stores the evaluation metrics.

        Args:
            start_index: The index of the first configuration to evaluate.
                         Allows resuming interrupted runs.
            target_index: If provided, only this specific configuration index
                          will be executed.
        """
        import datetime

        import src.utils as utils

        hyperparameter_combinations = generate_hyperparameter_combinations(
            self.model_config
        )

        # Determine all seeds
        seeds = (
            self.random_state
            if isinstance(self.random_state, list)
            else [self.random_state]
        )

        # Generate a flat list of runs: (combo_idx, model_config, varied_params, seed)
        all_runs = []
        for combo_idx, (model_config, varied_params) in enumerate(
            hyperparameter_combinations
        ):
            for seed in seeds:
                all_runs.append((combo_idx, model_config, varied_params, seed))

        num_runs = len(all_runs)

        if target_index is not None:
            if target_index < 0 or target_index >= num_runs:
                self.logger.error(
                    f"Target index {target_index + 1} is out of range (1-{num_runs})."
                )
                return
            run_runs = [all_runs[target_index]]
            self.logger.info(
                f"Running specific model configuration index {target_index + 1} "
                f"of {num_runs}."
            )
        else:
            if start_index >= num_runs:
                self.logger.info(
                    f"Start index {start_index} is beyond total combinations "
                    f"{num_runs}. Nothing to do."
                )
                return

            run_runs = all_runs[start_index:]
            if start_index == 0:
                self.logger.info(
                    f"Starting hyperparameter optimization for {num_runs} models."
                )
            else:
                self.logger.info(
                    "Resuming hyperparameter optimization for "
                    f"{num_runs} models "
                    f"(starting at index {start_index + 1})."
                )

        # New Metadata Capture
        start_timestamp = datetime.datetime.now().isoformat()
        dataset_name = pathlib.Path(
            self.experiment_config["experiment"]["dataset_path"]
        ).stem.replace("_embeddings", "")
        n_observations = len(self.texts)

        try:
            for run_idx, (combo_idx, model_config, varied_params, seed) in enumerate(
                run_runs
            ):
                model_id = self.model_config.get("id", "model")
                if len(seeds) > 1:
                    run_id = f"{model_id}_{combo_idx + 1}_seed{seed}"
                else:
                    run_id = f"{model_id}_{combo_idx + 1}"

                # Clean up param names for reporting
                cleaned_varied_params = clean_varied_params(varied_params)
                self.logger.info(f"--- Training model [{run_id}] with seed {seed} ---")
                self.logger.info(f"Varied Parameters: {cleaned_varied_params}")

                # 1. Create Model Instance
                topic_model = models.create_bertopic_instance(
                    model_config=model_config,
                    scaled_metadata=self.scaled_metadata,
                    random_state=seed,
                )

                # 2. Train and Evaluate
                try:
                    metrics, trained_model = training.train_and_evaluate(
                        topic_model=topic_model,
                        model_id=run_id,
                        text=self.texts,
                        embeddings=self.embeddings,
                        config=self.experiment_config,
                    )

                    # 3. Store results, including the varied hyperparameters
                    # and metadata
                    run_metadata = {
                        "experiment_id": self.experiment_id,
                        "random_state": seed,
                        "clustering_algo": model_config["clustering"]["type"],
                        "dim_red_algo": model_config["dimensionality_reduction"][
                            "type"
                        ],
                        "n_observations": n_observations,
                        "timestamp": start_timestamp,
                        "file_timestamp": self.file_timestamp,
                        "dataset_name": dataset_name,
                    }
                    metrics.update(run_metadata)
                    metrics.update(cleaned_varied_params)
                    self.results.append(metrics)

                    # 4. Extract Qualitative Data
                    qual_metadata = run_metadata.copy()
                    qual_metadata.update(cleaned_varied_params)
                    qual_df = utils.extract_qualitative_data(
                        trained_model, run_id, qual_metadata
                    )
                    self.qualitative_results.append(qual_df)

                except Exception as e:
                    self.logger.error(
                        f"Failed to train model [{run_id}] with params "
                        f"{cleaned_varied_params} and seed {seed}: {e}"
                    )
                    continue
        except KeyboardInterrupt:
            self.logger.warning(
                "Optimization interrupted by user. Cleaning up and saving results..."
            )

        self.logger.info("Finished hyperparameter optimization phase.")

    def save_results(
        self, filepath: str | pathlib.Path, decimal_digits: int | None = None
    ) -> None:
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

        # Calculated metrics from the experiment config
        exp_metrics = []
        if "experiment" in self.experiment_config:
            exp_metrics.extend(
                self.experiment_config["experiment"].get("coherence_metrics", [])
            )
            exp_metrics.extend(
                self.experiment_config["experiment"].get("diversity_metrics", [])
            )

        # Varied parameter columns are what's left over
        param_cols = [
            col
            for col in all_cols
            if col not in core_stats_cols and col not in exp_metrics
        ]

        # New order: core stats, then params, then calculated metrics
        final_order = (
            [c for c in core_stats_cols if c in all_cols]
            + [p for p in param_cols if p in all_cols]
            + [m for m in exp_metrics if m in all_cols]
        )
        df = df.select(final_order)

        # Handle appending/merging if the file already exists
        save_path = pathlib.Path(filepath)
        if save_path.exists():
            try:
                existing_df = pl.read_csv(save_path, infer_schema_length=None)
                # Ensure we cast new results to match the existing schema
                # for safe vertical concatenation
                df = pl.concat(
                    [existing_df, df.cast(existing_df.schema)], how="vertical"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to merge with existing results file: {e}. "
                    "Saving to a new file with suffix."
                )
                save_path = save_path.with_name(
                    f"{save_path.stem}_v2{save_path.suffix}"
                )

        df.write_csv(save_path, float_precision=decimal_digits)
        self.logger.info(f"Results saved to {save_path}")

        # Save Qualitative Data
        if self.qualitative_results:
            consolidated_qual_df = pl.concat(self.qualitative_results, how="diagonal")
            output_dir = save_path.parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{save_path.stem}.json"

            # Handle merging if the qualitative file already exists
            if output_path.exists():
                try:
                    existing_qual_df = pl.read_json(
                        output_path, infer_schema_length=None
                    )
                    consolidated_qual_df = pl.concat(
                        [
                            existing_qual_df,
                            consolidated_qual_df.cast(existing_qual_df.schema),
                        ],
                        how="vertical",
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to merge with existing qualitative results file: {e}"
                    )

            # Serialize to JSON string and then pretty-print using the
            # standard json library
            json_str = consolidated_qual_df.write_json()
            parsed_json = json.loads(json_str)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4)

            self.logger.info(f"Qualitative topic data saved at {output_path}")
