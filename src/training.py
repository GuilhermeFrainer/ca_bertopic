import numpy as np
from bertopic import BERTopic

from typing import Optional
import time
import logging

import src.evaluation as evaluation
import src.models as models


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
    umap_model = models.get_algorithm(
        model_config["dimensionality_reduction"],
        metadata=scaled_metadata,
        random_state=random_state
    )
    hdbscan_model = models.get_algorithm(
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
    
    octis_output = evaluation.bertopic_output_to_octis(topic_model, topics)

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
        metrics[cm] = evaluation.compute_coherence(
            model_output=octis_output,
            texts=tokenized_texts,
            measure=cm
        )

    # Diversity Loop
    for dm in config["experiment"]["diversity_metrics"]:
        metrics[dm] = evaluation.compute_diversity(dm, model_output=octis_output)

    return metrics, topic_model

