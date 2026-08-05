import numpy as np
from hdbscan import HDBSCAN
from umap import UMAP

from src.models import create_bertopic_instance, get_algorithm


def test_get_algorithm_hdbscan_defaults():
    """
    Tests that get_algorithm correctly falls back to defaults for HDBSCAN
    when 'params' is null/None or omitted.
    """
    # 1. Test when params is explicitly None (null in YAML)
    config_null = {"type": "hdbscan", "params": None}
    algo_null = get_algorithm(config_null, metadata=None, random_state=42)

    assert isinstance(algo_null, HDBSCAN)
    # Default min_cluster_size in HDBSCAN is 5
    assert algo_null.min_cluster_size == 5

    # 2. Test when params is omitted entirely
    config_omitted = {"type": "hdbscan"}
    algo_omitted = get_algorithm(config_omitted, metadata=None, random_state=42)

    assert isinstance(algo_omitted, HDBSCAN)
    assert algo_omitted.min_cluster_size == 5


def test_get_algorithm_umap_defaults():
    """
    Tests that get_algorithm correctly falls back to defaults for UMAP
    when 'params' is null/None or omitted (except for random_state
    which is passed explicitly).
    """
    config_null = {"type": "umap", "params": None}
    algo_null = get_algorithm(config_null, metadata=None, random_state=42)

    assert isinstance(algo_null, UMAP)
    # Default n_neighbors in UMAP is 15
    assert algo_null.n_neighbors == 15
    assert algo_null.random_state == 42


def test_create_bertopic_instance_custom_params():
    """
    Tests that create_bertopic_instance correctly parses bertopic
    params (like nr_topics).
    """
    model_config = {
        "dimensionality_reduction": {
            "type": "umap",
            "params": {"min_dist": 0.0, "metric": "cosine"},
        },
        "clustering": {"type": "hdbscan", "params": None},
        "bertopic": {"params": {"nr_topics": 20, "top_n_words": 10}},
    }

    # Mock metadata
    scaled_metadata = np.random.rand(10, 2)

    topic_model = create_bertopic_instance(
        model_config=model_config, scaled_metadata=scaled_metadata, random_state=42
    )

    assert topic_model.nr_topics == 20
    assert topic_model.top_n_words == 10


def test_create_bertopic_instance_remove_rep_stopwords():
    """
    Tests that remove_rep_stopwords correctly configures CountVectorizer
    with English stop words on BERTopic topic representations.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    model_config = {
        "dimensionality_reduction": {"type": "umap"},
        "clustering": {"type": "hdbscan"},
        "bertopic": {"params": {}},
    }
    scaled_metadata = np.random.rand(10, 2)

    # 1. Default (False) -> vectorizer_model is standard BERTopic default
    model_default = create_bertopic_instance(
        model_config=model_config, scaled_metadata=scaled_metadata, random_state=42
    )
    assert (
        not isinstance(model_default.vectorizer_model, CountVectorizer)
        or model_default.vectorizer_model.stop_words is None
    )

    # 2. Flag set via argument
    model_flag = create_bertopic_instance(
        model_config=model_config,
        scaled_metadata=scaled_metadata,
        random_state=42,
        remove_rep_stopwords=True,
    )
    assert isinstance(model_flag.vectorizer_model, CountVectorizer)
    assert model_flag.vectorizer_model.stop_words == "english"

    # 3. Flag set via config dict
    config_with_stopwords = {
        "dimensionality_reduction": {"type": "umap"},
        "clustering": {"type": "hdbscan"},
        "bertopic": {"params": {"remove_rep_stopwords": True}},
    }
    model_cfg = create_bertopic_instance(
        model_config=config_with_stopwords,
        scaled_metadata=scaled_metadata,
        random_state=42,
    )
    assert isinstance(model_cfg.vectorizer_model, CountVectorizer)
    assert model_cfg.vectorizer_model.stop_words == "english"
