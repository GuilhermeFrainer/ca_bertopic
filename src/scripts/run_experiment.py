import polars as pl
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mvlearn.cluster import MultiviewSpectralClustering
from bertopic import BERTopic
from tqdm import tqdm
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO

from typing import Optional
from pathlib import Path
import yaml
import datetime
import argparse
import time
import logging

from src.mvc_wrapper import MVCWrapper
from src.evaluation import bertopic_output_to_octis


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
        config = load_config(args.exp)
        exp_name = config["experiment"]["name"]
        logger = setup_logging(exp_name)
        
        logger.info("Loading and preparing data...")
        text, embeddings, scaled_metadata = load_and_prep_data(config)

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
            metrics, trained_model = train_and_evaluate(
                baseline_config, text, embeddings, scaled_metadata, config
            )
            results.append(metrics)
            
            # Extract n_topics
            baseline_n_topics = len(trained_model.get_topic_info()) - 1
            logger.info(f"Baseline found {baseline_n_topics} topics.")

        # 4. Run Remaining Models
        for model_config in tqdm(other_models, desc="Training models"):
            try:
                metrics, _ = train_and_evaluate(
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


def load_config(exp_name: str) -> dict:
    """
    Resolves path and loads the YAML config
    """
    logger = logging.getLogger("pipeline")

    filename = exp_name if exp_name.endswith(".yaml") else f"{exp_name}.yaml"
    config_path = EXPERIMENTS_DIR / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file {config_path} not found.")

    with open(config_path, "r") as f:
        logger.info(f"Loaded config from {config_path}")
        return yaml.safe_load(f)
    

def load_and_prep_data(config: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Loads parquet, samples data, and scales metadata.
    """
    data_path = config["experiment"]["dataset_path"]
    sample_size = config["experiment"]["sample_size"]
    random_state = config["experiment"]["random_state"]
    covariates = config["experiment"]["covariates"]

    # Lazy load and sample
    full_lf = pl.scan_parquet(data_path)
    lf = sample_from_lf(full_lf, n=sample_size, seed=random_state)
    
    # Materialize data
    df = lf.collect()
    
    text = df["text"].to_list()
    embeddings = df["embedding"].to_numpy()
    
    # Metadata scaling
    metadata_df = df.select(covariates)
    scaling_expressions = [min_max_scaler(c) for c in metadata_df.columns]
    scaled_metadata = metadata_df.with_columns(scaling_expressions).to_numpy()

    return text, embeddings, scaled_metadata


def train_and_evaluate(
    model_config: dict,
    text: list[str],
    embeddings: np.ndarray,
    scaled_metadata: np.ndarray,
    config: dict,
    baseline_topics: Optional[int] = None
) -> tuple[dict, BERTopic]:
    """
    Trains a single model instance and returns metrics.
    """
    start_time = time.time()
    logger = logging.getLogger("pipeline")

    random_state = config["experiment"]["random_state"]
    model_name = model_config.get("id", "Unnamed Model")

    # 1. Instantiation
    umap_model = get_algorithm(
        model_config["dimensionality_reduction"],
        metadata=scaled_metadata,
        random_state=random_state
    )
    hdbscan_model = get_algorithm(
        model_config["clustering"],
        metadata=scaled_metadata,
        random_state=random_state,
        n_clusters=baseline_topics # Only used if provided
    )
    
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
    topics, probs = topic_model.fit_transform(documents=text, embeddings=embeddings)

    # 2. Basic Metrics
    outlier_count = topics.count(-1)
    n_topics = len(topic_model.get_topic_info()) - 1

    # 3. Advanced Metrics (Coherence & Diversity)
    # Pre-tokenize once per model run (or pass in pre-tokenized text to save time)
    analyzer = topic_model.vectorizer_model.build_analyzer()
    tokenized_texts = [analyzer(t) for t in text]
    
    octis_output = bertopic_output_to_octis(topic_model, topics)

    duration = time.time() - start_time
    logger.info(f"[{model_name}] Finished in {duration:.2f} seconds.")
    
    metrics = {
        "model_name": model_name,
        "duration_seconds": duration,
        "n_topics": n_topics,
        "outliers": outlier_count
    }

    # Coherence Loop
    for cm in config["experiment"]["coherence_metrics"]:
        coherence_model = Coherence(texts=tokenized_texts, measure=cm)
        metrics[cm] = coherence_model.score(model_output=octis_output)

    # Diversity Loop
    for dm in config["experiment"]["diversity_metrics"]:
        metrics[dm] = compute_diversity(dm, model_output=octis_output)

    return metrics, topic_model


def setup_logging(experiment_name: str) -> logging.Logger:
    """
    Sets up a logger that writes to file and console.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = LOG_DIR / f"{experiment_name}-{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Keeps printing to console
        ]
    )
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.info(f"Starting experiment: {experiment_name}")
    return logger


def compute_diversity(diversity_type: str, model_output: dict) -> float:
    if diversity_type == "irbo":
        diversity_model = InvertedRBO()
    elif diversity_type == "topic_diversity":
        diversity_model = TopicDiversity()
    else:
        raise ValueError(f"Invalid diversity type: {diversity_type}")
    return diversity_model.score(model_output=model_output) # type: ignore


def min_max_scaler(col: str):
    x = pl.col(col)
    return (x - x.min()) / (x.max() - x.min())


def get_algorithm(
    config: dict,
    metadata: Optional[np.ndarray],
    random_state: int,
    n_clusters: Optional[int] = None
):
    algo_type = config["type"]
    params = config.get("params") or {}

    if algo_type == 'umap':
        return UMAP(random_state=random_state, **params)
    elif algo_type == 'pca':
        return PCA(random_state=random_state, **params)
    elif algo_type == 'hdbscan':
        return HDBSCAN(**params) # No random state parameter
    elif algo_type == 'k_means':
        return KMeans(random_state=random_state, **params)
    elif algo_type == 'multi_view_spectral_clustering':
        if metadata is None:
            raise ValueError("Metadata array is null")

        if n_clusters and params["n_clusters"] == "baseline":
            params["n_clusters"] = n_clusters

        cluster_model = MultiviewSpectralClustering(
            random_state=random_state,
            **params
        )
        return MVCWrapper(model=cluster_model, metadata=metadata)
        
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
    

def sample_from_lf(
    lf: pl.LazyFrame,
    n: int,
    seed: Optional[int] = None,
    replace: bool = False
) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    lf_len = lf.select("index").count().collect().item()
    all_possible_rows = np.arange(lf_len)
    sample_idxs = rng.choice(all_possible_rows, size=n, replace=replace)
    return lf.filter(pl.col("index").is_in(sample_idxs))


if __name__ == "__main__":
    main()

