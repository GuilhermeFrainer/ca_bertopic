import logging
import time
from typing import Any, Optional

import numpy as np
from tritopic import TriTopic

import src.evaluation as evaluation


def train_and_evaluate(
    topic_model: Any,
    model_id: str,
    text: list[str],
    embeddings: np.ndarray,
    config: dict,
    scaled_metadata: Optional[np.ndarray] = None,
) -> tuple[dict, Any]:
    """
    Fits a pre-instantiated topic model (BERTopic or TriTopic) and calculates evaluation metrics.

    Args:
        topic_model: The instantiated BERTopic or TriTopic object.
        model_id: String identifier for logging/results.
        text: List of document strings.
        embeddings: Pre-computed document embeddings.
        config: The global experiment configuration (for metric settings).
        scaled_metadata: Optional pre-processed metadata matrix for multi-view / tritopic models.

    Returns:
        A tuple containing the metrics dictionary and the fitted model.
    """
    logger = logging.getLogger("pipeline")

    start_time = time.time()
    is_tritopic = isinstance(topic_model, TriTopic)

    if is_tritopic:
        # Check if metadata should be used
        if scaled_metadata is not None:
            topic_model.fit(
                documents=text, embeddings=embeddings, metadata=scaled_metadata
            )
        else:
            topic_model.fit(documents=text, embeddings=embeddings)

        labels = getattr(topic_model, "labels_", [])
        outlier_count = int((np.array(labels) == -1).sum()) if len(labels) > 0 else 0
        n_topics = len([t for t in topic_model.topics_ if t.topic_id != -1])
    else:
        topics, _ = topic_model.fit_transform(documents=text, embeddings=embeddings)
        outlier_count = topics.count(-1) if -1 in topics else 0
        n_topics = len([t for t in topic_model.get_topics() if t != -1])

    duration = time.time() - start_time
    logger.info(f"[{model_id}] Training finished in {duration:.2f} seconds.")

    # Build tokenized texts for OCTIS metrics
    if hasattr(topic_model, "vectorizer_model") and hasattr(
        topic_model.vectorizer_model, "build_analyzer"
    ):
        analyzer = topic_model.vectorizer_model.build_analyzer()
        tokenized_texts = [analyzer(t) for t in text]
    else:
        tokenized_texts = [t.lower().split() for t in text]

    tokenized_texts = [t for t in tokenized_texts if len(t) > 0]

    if is_tritopic:
        octis_output = evaluation.tritopic_output_to_octis(topic_model)
    else:
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
