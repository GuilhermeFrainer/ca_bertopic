import logging
import time

import numpy as np
from bertopic import BERTopic

import src.evaluation as evaluation


def train_and_evaluate(
    topic_model: BERTopic,
    model_id: str,
    text: list[str],
    embeddings: np.ndarray,
    config: dict,
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

    # 1. Fit & Transform
    start_time = time.time()
    topics, _ = topic_model.fit_transform(documents=text, embeddings=embeddings)
    duration = time.time() - start_time
    logger.info(f"[{model_id}] Training finished in {duration:.2f} seconds.")

    # 2. Basic Metrics
    outlier_count = topics.count(-1) if -1 in topics else 0
    # Count topics >= 0
    n_topics = len([t for t in topic_model.get_topics() if t != -1])

    # 3. Advanced Metrics (Coherence & Diversity)
    # Optimization: Re-use vectorizer analyzer
    analyzer = topic_model.vectorizer_model.build_analyzer()
    tokenized_texts = [analyzer(t) for t in text]

    # Filter out empty tokenized documents as they can break some coherence metrics
    tokenized_texts = [t for t in tokenized_texts if len(t) > 0]

    octis_output = evaluation.bertopic_output_to_octis(topic_model)

    metrics = {
        "model_name": model_id,
        "duration_seconds": duration,
        "n_topics": n_topics,
        "outliers": outlier_count,
    }

    # Coherence Loop
    for cm in config["experiment"]["coherence_metrics"]:
        metrics[cm] = evaluation.compute_coherence(
            model_output=octis_output, texts=tokenized_texts, measure=cm
        )

    # Diversity Loop
    for dm in config["experiment"]["diversity_metrics"]:
        metrics[dm] = evaluation.compute_diversity(dm, model_output=octis_output)

    return metrics, topic_model
