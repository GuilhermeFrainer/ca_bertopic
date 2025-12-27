from bertopic import BERTopic
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


def bertopic_output_to_octis(
    m: BERTopic,
    topic_assignments: list[int],
    topk: int = 10
) -> dict[str, list[list[str]]]:
    """
    Reshapes BERTopic output so that it can be readily passed to OCTIS
    for evaluation.
    """
    topic_words: list[list[str]] = []
    # Excludes noise topic -1
    n_topics = len(set(topic_assignments)) - 1
    for i in range(n_topics):
        topic_info = m.get_topic(i)
        if isinstance(topic_info, list):
            words = [word for word, _ in topic_info[:topk]] # type: ignore
            topic_words.append(words)
    return {"topics": topic_words}


def compute_coherence(
    model_output: dict,
    texts: list[list[str]],
    measure: str = "c_npmi",
    topk: int = 10
) -> float:
    coherence_model = Coherence(
        texts=texts, 
        topk=topk,
        measure=measure
    )
    return coherence_model.score(model_output)

