import logging
import random

import numpy as np
from bertopic import BERTopic
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import InvertedRBO, TopicDiversity


def topic_words_to_octis(topic_words: list[list[str]]) -> dict[str, list[list[str]]]:
    """
    Standardizes a list of topic words into OCTIS format.
    """
    return {"topics": topic_words}


def get_top_words_from_beta(
    beta: np.ndarray, vocab: list[str], topk: int = 10
) -> list[list[str]]:
    """
    Extracts the top k words for each topic from a beta matrix (K x V).
    """
    topic_words = []
    for topic_beta in beta:
        # Assuming beta is either probabilities or log-probabilities
        top_indices = np.argsort(topic_beta)[::-1][:topk]
        words = [vocab[i] for i in top_indices]
        topic_words.append(words)
    return topic_words


def bertopic_output_to_octis(m: BERTopic, topk: int = 10) -> dict[str, list[list[str]]]:
    """
    Reshapes BERTopic output so that it can be readily passed to OCTIS
    for evaluation.
    """
    topic_words: list[list[str]] = []
    topic_ids = [
        t_id
        for t_id in m.get_topics().keys()
        if t_id != -1  # Ignores noise topic
    ]
    for t_id in topic_ids:
        topic_info = m.get_topic(t_id)  # type: ignore
        if isinstance(topic_info, list):
            # 1. Filter all available words first
            words = [
                str(word).strip()
                for word, _ in topic_info
                if str(word).strip() != "" and not str(word).strip().isdigit()
            ]

            # 2. Slice to topk
            words = words[:topk]

            # 3. If we STILL have less than topk, log it and pad by sampling
            if 0 < len(words) < topk:
                logger = logging.getLogger("pipeline")
                logger.warning(
                    f"Topic {t_id} only has {len(words)} words after filtering "
                    f"(requested {topk}). Padding by sampling from existing words."
                )
                words.extend(random.choices(words, k=topk - len(words)))

            topic_words.append(words)

    return {"topics": topic_words}


def compute_coherence(
    model_output: dict, texts: list[list[str]], measure: str = "c_npmi", topk: int = 10
) -> float:
    logger = logging.getLogger("pipeline")

    # Check if there are any topics to evaluate
    if not model_output.get("topics"):
        logger.warning(f"No topics found for evaluation of {measure}. Returning 0.0")
        return 0.0

    # Ensure all topic words are present in the dictionary/texts
    # and that topics are not empty lists.
    filtered_topics = [t for t in model_output["topics"] if len(t) > 0]
    if len(filtered_topics) < len(model_output["topics"]):
        logger.warning(
            f"Removed {len(model_output['topics']) - len(filtered_topics)} "
            "empty topics."
        )

    if not filtered_topics:
        logger.warning(
            f"No non-empty topics found for evaluation of {measure}. Returning 0.0"
        )
        return 0.0

    model_output["topics"] = filtered_topics

    coherence_model = Coherence(texts=texts, topk=topk, measure=measure)
    try:
        return coherence_model.score(model_output)
    except IndexError as e:
        logger.error(
            f"Error when computing coherence ({measure}). "
            f"Model output topics:\n{model_output['topics']}"
        )
        raise e
    except ValueError as e:
        logger.error(
            f"Error when computing coherence ({measure}). "
            f"Model output topics:\n{model_output['topics']}"
        )
        # Log a few examples of texts to help debugging without flooding the log
        if texts:
            logger.error(f"First 5 tokenized texts: {texts[:5]}")
        raise e


def compute_diversity(diversity_type: str, model_output: dict) -> float:
    if diversity_type == "irbo":
        diversity_model = InvertedRBO()
    elif diversity_type == "topic_diversity":
        diversity_model = TopicDiversity()
    else:
        raise ValueError(f"Invalid diversity type: {diversity_type}")
    return diversity_model.score(model_output=model_output)  # type: ignore
