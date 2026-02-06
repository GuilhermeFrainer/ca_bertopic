import numpy as np
from bertopic import BERTopic

from typing import Optional
import time
import logging

import src.evaluation as evaluation
import src.models as models


def train_and_evaluate(
    topic_model: BERTopic,
    model_id: str,
    text: list[str],
    embeddings: np.ndarray,
    config: dict
) -> tuple[dict, BERTopic]:
    """
    Fits a pre-instantiated BERTopic model and calculates evaluation metrics.

    Args:
        topic_model: The instantiated BERTopic object.
        model_id: String identifier for logging/results.
        text: List of document strings.
        embeddings: Pre-computed document embeddings.
        config: The global experiment configuration (for metric settings).

    Returns:
        A tuple containing the metrics dictionary and the fitted model.
    """
    logger = logging.getLogger("pipeline")
    start_time = time.time()

    # 1. Fit & Transform
    # The model is already instantiated, so we just fit it.
    topics, _probs = topic_model.fit_transform(documents=text, embeddings=embeddings)

    # 2. Basic Metrics
    outlier_count = topics.count(-1)
    # n_topics is length of info minus the outlier topic (-1)
    n_topics = len(topic_model.get_topic_info()) - 1

    # 3. Advanced Metrics (Coherence & Diversity)
    # Optimization: Re-use vectorizer analyzer
    analyzer = topic_model.vectorizer_model.build_analyzer()
    tokenized_texts = [analyzer(t) for t in text]
    
    octis_output = evaluation.bertopic_output_to_octis(topic_model)

    duration = time.time() - start_time
    logger.info(f"[{model_id}] Finished in {duration:.2f} seconds.")
    
    metrics = {
        "model_name": model_id,
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

