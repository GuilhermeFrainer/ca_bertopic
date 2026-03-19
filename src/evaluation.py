from bertopic import BERTopic
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.coherence_metrics import Coherence

import logging


def bertopic_output_to_octis(
    m: BERTopic,
    topk: int = 10
) -> dict[str, list[list[str]]]:
    """
    Reshapes BERTopic output so that it can be readily passed to OCTIS
    for evaluation.
    """
    topic_words: list[list[str]] = []
    topic_ids = [
        t_id
        for t_id in m.get_topics().keys() 
        if t_id != -1 # Ignores noise topic
    ]
    for t_id in topic_ids:
        topic_info = m.get_topic(t_id) # type: ignore
        if isinstance(topic_info, list):
            words = [
                str(word).strip() 
                for word, _ in topic_info[:topk] 
                if str(word).strip() != "" and not str(word).strip().isdigit()
            ]
            topic_words.append(words)

    return {"topics": topic_words}


def compute_coherence(
    model_output: dict,
    texts: list[list[str]],
    measure: str = "c_npmi",
    topk: int = 10
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
        logger.warning(f"Removed {len(model_output['topics']) - len(filtered_topics)} empty topics.")
    
    if not filtered_topics:
        logger.warning(f"No non-empty topics found for evaluation of {measure}. Returning 0.0")
        return 0.0
    
    model_output["topics"] = filtered_topics

    coherence_model = Coherence(
        texts=texts, 
        topk=topk,
        measure=measure
    )
    try:
        return coherence_model.score(model_output)
    except IndexError as e:
        logger.error(f"Error when computing coherence ({measure}). Model output topics:\n{model_output['topics']}")
        raise e
    except ValueError as e:
        logger.error(f"Error when computing coherence ({measure}). Model output topics:\n{model_output['topics']}")
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
    return diversity_model.score(model_output=model_output) # type: ignore

