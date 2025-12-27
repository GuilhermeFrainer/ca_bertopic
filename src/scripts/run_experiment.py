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

from src.mvc_wrapper import MVCWrapper
from src.evaluation import bertopic_output_to_octis


EXPERIMENTS_DIR = pathlib.Path("experiments")
OUTPUT_DIR = pathlib.Path("output")


def main():
    now = datetime.datetime.now()
    with open(EXPERIMENTS_DIR / "experiment_1.yaml", "r") as f:
        config = yaml.safe_load(f)

    random_state = config["experiment"]["random_state"]
    sample_size = config["experiment"]["sample_size"]
    covariates = config["experiment"]["covariates"]
    data_path = config["experiment"]["dataset_path"]

    models_config: list[dict] = config["models"]

    # Get the data
    full_lf = pl.scan_parquet(data_path)
    lf = sample_from_lf(full_lf, n=sample_size, seed=random_state)
    
    text = lf.select("text").collect().to_series().to_list()
    embeddings = lf.select("embedding").collect().to_series().to_numpy()
    metadata_df = lf.select(covariates).collect()

    scaling_expressions = [
        min_max_scaler(c)
        for c in metadata_df.columns
    ]

    scaled_metadata = metadata_df.with_columns(scaling_expressions).to_numpy()

    results = []
    for model_config in tqdm(models_config, desc="Training models"):
        #print(model_config)
        model_name = model_config.get("id", "Unnamed Model")
        try:
            # Model instantiation
            umap_model = get_algorithm(
                model_config["dimensionality_reduction"],
                metadata=scaled_metadata,
                random_state=random_state
            )
            hdbscan_model = get_algorithm(
                model_config["clustering"],
                metadata=scaled_metadata,
                random_state=random_state
            )
            topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
            topics, probs = topic_model.fit_transform(documents=text, embeddings=embeddings)

            # Evaluation
            outlier_count = topics.count(-1)

            # Coherence
            # Texts have to be tokenized to compute coherence
            analyzer = topic_model.vectorizer_model.build_analyzer()
            tokenized_texts = [analyzer(t) for t in text]
            octis_output = bertopic_output_to_octis(
                topic_model,
                topics
            )
            coherence_scores = {}
            for cm in config["experiment"]["coherence_metrics"]:
                coherence_model = Coherence(
                    texts=tokenized_texts,
                    measure=cm
                )
                coherence_scores[cm] = coherence_model.score(model_output=octis_output)

            # Diversity
            diversity_scores = {}
            for dm in config["experiment"]["diversity_metrics"]:
                diversity_scores[dm] = compute_diversity(dm, model_output=octis_output)

            run_metrics = {
                "model_name": model_name,
                "n_topics": len(topic_model.get_topic_info()) - 1,
                "outliers": outlier_count
            }
            results.append(run_metrics | coherence_scores | diversity_scores)
        except Exception as e:
            print(f"Error training model {model_name}: {e}")
            raise e
            results.append({
                "model_name": model_name,
                "error": str(e)
            })
    
    results_df = pl.DataFrame(results)
    experiment_name = config["experiment"]["name"]
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    results_path = OUTPUT_DIR / f"{experiment_name}-{timestamp}.csv"
    results_df.write_csv(results_path)
    print(f"Results saved to {results_path}")


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
    random_state: int
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

