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
import pathlib
import yaml
import datetime
import argparse

from src.mvc_wrapper import MVCWrapper
from src.evaluation import bertopic_output_to_octis


EXPERIMENTS_DIR = pathlib.Path("experiments")
OUTPUT_DIR = pathlib.Path("output")


def main():
    parser = argparse.ArgumentParser(description="Run a specific ML experiment.")
    parser.add_argument(
        "--exp", 
        type=str, 
        required=True, 
        help="Name of the experiment yaml file (e.g., experiment_2)"
    )
    args = parser.parse_args()
    now = datetime.datetime.now()

    try:
        # 2. Setup
        config = load_config(args.exp)
        text, embeddings, scaled_metadata = load_and_prep_data(config)
        
        # 3. Main Experiment Loop
        results = []
        baseline_topic_n = None
        models_config = config["models"]

        for model_config in tqdm(models_config, desc="Training models"):
            try:
                # Pass baseline_topic_n if we have it, logic handled inside or here
                metrics, trained_model = train_and_evaluate(
                    model_config, 
                    text, 
                    embeddings, 
                    scaled_metadata, 
                    config,
                    baseline_topics=baseline_topic_n
                )
                results.append(metrics)

                # Capture baseline for subsequent models
                if metrics["model_name"] == "vanilla":
                    baseline_topic_n = len(trained_model.get_topic_info())

            except Exception as e:
                print(f"Error training model {model_config.get('id')}: {e}")
                # decide if you want to 'continue' or 'raise' here
                raise e

        # 4. Save Results
        results_df = pl.DataFrame(results)
        experiment_name = config["experiment"]["name"]
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True) # Safety check
        results_path = OUTPUT_DIR / f"{experiment_name}-{timestamp}.csv"
        results_df.write_csv(results_path)
        print(f"Results saved to {results_path}")

    except Exception as e:
        print(f"Pipeline failed: {e}")


def load_config(exp_name: str) -> dict:
    """
    Resolves path and loads the YAML config
    """
    filename = exp_name if exp_name.endswith(".yaml") else f"{exp_name}.yaml"
    config_path = EXPERIMENTS_DIR / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file {config_path} not found.")

    with open(config_path, "r") as f:
        print(f"Loaded config from {config_path}")
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
    
    metrics = {
        "model_name": model_name,
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

