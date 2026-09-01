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


def test_create_tritopic_instance_defaults():
    """
    Tests that create_tritopic_instance sets expected default parameters
    such as use_metadata_view=True and random_state.
    """
    from tritopic import TriTopic

    from src.models import create_topic_model_instance, create_tritopic_instance

    model_config = {"type": "tritopic", "params": {}}
    model = create_tritopic_instance(model_config=model_config, random_state=42)

    assert isinstance(model, TriTopic)
    assert model.config.use_metadata_view is True
    assert model.config.random_state == 42
    assert model.n_topics == "auto"

    # Test via factory function routing
    factory_model = create_topic_model_instance(
        model_config=model_config, scaled_metadata=None, random_state=123
    )
    assert isinstance(factory_model, TriTopic)
    assert factory_model.config.random_state == 123


def test_create_tritopic_instance_custom_params():
    """
    Tests that custom parameters passed in YAML model config (e.g. n_neighbors,
    n_topics, use_metadata_view) are correctly set on TriTopic and TriTopicConfig.
    """
    from src.models import create_tritopic_instance

    model_config = {
        "type": "tritopic",
        "params": {
            "n_neighbors": 25,
            "n_topics": 15,
            "use_metadata_view": False,
            "verbose": True,
        },
    }
    model = create_tritopic_instance(model_config=model_config, random_state=42)

    assert model.config.n_neighbors == 25
    assert model.n_topics == 15
    assert model.config.use_metadata_view is False
    assert model.config.verbose is True


def test_create_tritopic_instance_n_clusters_override():
    """
    Tests that n_clusters passed as argument (e.g. from baseline) or in params
    is properly mapped to TriTopic's n_topics parameter.
    """
    from src.models import create_tritopic_instance

    # 1. Explicit n_clusters argument (e.g. from run_experiment matching baseline)
    model_config_1 = {"type": "tritopic", "params": {}}
    model_1 = create_tritopic_instance(
        model_config=model_config_1, random_state=42, n_clusters=20
    )
    assert model_1.n_topics == 20

    # 2. n_clusters in config params (e.g. from standard grid sweep)
    model_config_2 = {"type": "tritopic", "params": {"n_clusters": 30}}
    model_2 = create_tritopic_instance(model_config=model_config_2, random_state=42)
    assert model_2.n_topics == 30


def test_create_fast_tritopic_instance_defaults():
    """
    Tests that create_fast_tritopic_instance sets expected default parameters
    such as use_metadata_view=True and random_state.
    """
    from fast_tritopic import FastTriTopic

    from src.models import create_fast_tritopic_instance, create_topic_model_instance

    model_config = {"type": "fast_tritopic", "params": {}}
    model = create_fast_tritopic_instance(model_config=model_config, random_state=42)

    assert isinstance(model, FastTriTopic)
    assert model.config.use_metadata_view is True
    assert model.config.random_state == 42
    assert model.n_topics == "auto"

    # Test via factory function routing
    factory_model = create_topic_model_instance(
        model_config=model_config, scaled_metadata=None, random_state=123
    )
    assert isinstance(factory_model, FastTriTopic)
    assert factory_model.config.random_state == 123


def test_create_fast_tritopic_instance_custom_params():
    """
    Tests that custom parameters passed in YAML model config (e.g. n_neighbors,
    n_topics, use_metadata_view) are correctly set on FastTriTopic.
    """
    from src.models import create_fast_tritopic_instance

    model_config = {
        "type": "fast_tritopic",
        "params": {
            "n_neighbors": 25,
            "n_topics": 15,
            "use_metadata_view": False,
            "verbose": True,
        },
    }
    model = create_fast_tritopic_instance(model_config=model_config, random_state=42)

    assert model.config.n_neighbors == 25
    assert model.n_topics == 15
    assert model.config.use_metadata_view is False
    assert model.config.verbose is True


def test_create_fast_tritopic_instance_n_clusters_override():
    """
    Tests that n_clusters passed as argument or in params
    is properly mapped to FastTriTopic's n_topics parameter.
    """
    from src.models import create_fast_tritopic_instance

    # 1. Explicit n_clusters argument
    model_config_1 = {"type": "fast_tritopic", "params": {}}
    model_1 = create_fast_tritopic_instance(
        model_config=model_config_1, random_state=42, n_clusters=20
    )
    assert model_1.n_topics == 20

    # 2. n_clusters in config params
    model_config_2 = {"type": "fast_tritopic", "params": {"n_clusters": 30}}
    model_2 = create_fast_tritopic_instance(
        model_config=model_config_2, random_state=42
    )
    assert model_2.n_topics == 30


def test_train_and_evaluate_bertopic_with_scaled_metadata():
    """
    Tests that train_and_evaluate correctly fits a BERTopic model without
    passing metadata to BERTopic.fit when scaled_metadata is provided.
    """
    from bertopic import BERTopic

    import src.training as training

    model_config = {
        "dimensionality_reduction": {"type": "umap"},
        "clustering": {"type": "hdbscan"},
    }
    scaled_metadata = np.random.rand(20, 2)
    topic_model = create_bertopic_instance(
        model_config=model_config, scaled_metadata=scaled_metadata, random_state=42
    )

    texts = [f"This is document number {i}" for i in range(20)]
    embeddings = np.random.rand(20, 10)
    config = {
        "experiment": {
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }

    metrics, trained = training.train_and_evaluate(
        topic_model=topic_model,
        model_id="test_bertopic",
        text=texts,
        embeddings=embeddings,
        config=config,
        scaled_metadata=scaled_metadata,
    )

    assert isinstance(trained, BERTopic)
    assert "n_topics" in metrics


def test_train_and_evaluate_tritopic_dispatches_metadata_polars_df():
    """
    Tests that train_and_evaluate correctly converts a Polars DataFrame
    metadata to pandas DataFrame when calling TriTopic.fit.
    """
    from unittest.mock import MagicMock

    import pandas as pd
    import polars as pl
    from tritopic import TriTopic

    import src.training as training

    mock_tritopic = MagicMock(spec=TriTopic)
    mock_tritopic.topics_ = []
    mock_tritopic.labels_ = []

    texts = ["doc a", "doc b"]
    embeddings = np.random.rand(2, 5)
    scaled_metadata = pl.DataFrame({"feature1": [0.1, 0.2], "feature2": [1.0, 0.0]})
    config = {
        "experiment": {
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }

    training.train_and_evaluate(
        topic_model=mock_tritopic,
        model_id="test_tritopic",
        text=texts,
        embeddings=embeddings,
        config=config,
        scaled_metadata=scaled_metadata,
    )

    mock_tritopic.fit.assert_called_once()
    call_kwargs = mock_tritopic.fit.call_args.kwargs
    assert "metadata" in call_kwargs
    assert isinstance(call_kwargs["metadata"], pd.DataFrame)
    assert list(call_kwargs["metadata"].columns) == ["feature1", "feature2"]


def test_train_and_evaluate_tritopic_dispatches_metadata_numpy_array():
    """
    Tests that train_and_evaluate converts a NumPy array metadata
    to a pandas DataFrame with columns when calling TriTopic.fit.
    """
    from unittest.mock import MagicMock

    import pandas as pd
    from tritopic import TriTopic

    import src.training as training

    mock_tritopic = MagicMock(spec=TriTopic)
    mock_tritopic.topics_ = []
    mock_tritopic.labels_ = []

    texts = ["doc a", "doc b"]
    embeddings = np.random.rand(2, 5)
    scaled_metadata = np.random.rand(2, 3)
    config = {
        "experiment": {
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }

    training.train_and_evaluate(
        topic_model=mock_tritopic,
        model_id="test_tritopic",
        text=texts,
        embeddings=embeddings,
        config=config,
        scaled_metadata=scaled_metadata,
    )

    mock_tritopic.fit.assert_called_once()
    call_kwargs = mock_tritopic.fit.call_args.kwargs
    assert "metadata" in call_kwargs
    assert isinstance(call_kwargs["metadata"], pd.DataFrame)
    assert call_kwargs["metadata"].shape == (2, 3)
    assert hasattr(call_kwargs["metadata"], "columns")


def test_train_and_evaluate_tritopic_empty_metadata():
    """
    Tests that train_and_evaluate passes metadata=None when scaled_metadata is empty.
    """
    from unittest.mock import MagicMock

    import polars as pl
    from tritopic import TriTopic

    import src.training as training

    mock_tritopic = MagicMock(spec=TriTopic)
    mock_tritopic.topics_ = []
    mock_tritopic.labels_ = []

    texts = ["doc a", "doc b"]
    embeddings = np.random.rand(2, 5)
    empty_metadata = pl.DataFrame()
    config = {
        "experiment": {
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }

    training.train_and_evaluate(
        topic_model=mock_tritopic,
        model_id="test_tritopic_empty",
        text=texts,
        embeddings=embeddings,
        config=config,
        scaled_metadata=empty_metadata,
    )

    mock_tritopic.fit.assert_called_once_with(documents=texts, embeddings=embeddings)


def test_train_and_evaluate_tritopic_real_execution():
    """
    Integration test verifying real TriTopic model fitting with Polars metadata
    without throwing AttributeError or crashing OCTIS metrics computation.
    """
    import polars as pl
    from tritopic import TriTopic, TriTopicConfig

    import src.training as training

    vocab = ["apple", "banana", "orange", "grape", "car", "train", "plane", "boat"]
    rng = np.random.default_rng(42)
    texts = [" ".join(rng.choice(vocab, size=8)) for _ in range(50)]
    embeddings = rng.random((50, 16))
    scaled_metadata = pl.DataFrame(
        {
            "meta_num1": rng.random(50),
            "meta_num2": rng.random(50),
        }
    )

    config_obj = TriTopicConfig(verbose=False, random_state=42)
    model = TriTopic(config=config_obj, n_topics=2)

    config = {
        "experiment": {
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }

    metrics, fitted_model = training.train_and_evaluate(
        topic_model=model,
        model_id="real_tritopic",
        text=texts,
        embeddings=embeddings,
        config=config,
        scaled_metadata=scaled_metadata,
    )

    assert "n_topics" in metrics
    assert metrics["n_topics"] > 0
    assert hasattr(fitted_model, "topics_")

    # Verify extract_qualitative_data maps doc indices to actual text strings
    import src.utils as utils

    qual_df = utils.extract_qualitative_data(
        fitted_model, "real_tritopic", {"dataset_name": "test_ds"}
    )
    assert "representative_docs" in qual_df.columns
    assert "representation" in qual_df.columns
    assert qual_df["representative_docs"].dtype == pl.List(pl.String)
    assert qual_df["representation"].dtype == pl.List(pl.String)

    rep_docs_sample = qual_df["representative_docs"].to_list()[0]
    assert len(rep_docs_sample) > 0
    # First document should be one of the original input texts, not a stringified int
    assert rep_docs_sample[0] in texts


def test_train_and_evaluate_fast_tritopic_real_execution():
    """
    Integration test verifying real FastTriTopic model fitting with Polars metadata
    and vectorized graph construction without crashing.
    """
    import polars as pl
    from fast_tritopic import FastTriTopic
    from tritopic import TriTopicConfig

    import src.training as training

    vocab = ["apple", "banana", "orange", "grape", "car", "train", "plane", "boat"]
    rng = np.random.default_rng(42)
    texts = [" ".join(rng.choice(vocab, size=8)) for _ in range(50)]
    embeddings = rng.random((50, 16))
    scaled_metadata = pl.DataFrame(
        {
            "meta_num1": rng.random(50),
            "meta_num2": rng.random(50),
            "meta_cat": ["A", "B", "A", "B", "C"] * 10,
        }
    )

    config_obj = TriTopicConfig(verbose=False, random_state=42)
    model = FastTriTopic(config=config_obj, n_topics=2)

    config = {
        "experiment": {
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }

    metrics, fitted_model = training.train_and_evaluate(
        topic_model=model,
        model_id="real_fast_tritopic",
        text=texts,
        embeddings=embeddings,
        config=config,
        scaled_metadata=scaled_metadata,
    )

    assert "n_topics" in metrics
    assert metrics["n_topics"] > 0
    assert hasattr(fitted_model, "topics_")

    import src.utils as utils

    qual_df = utils.extract_qualitative_data(
        fitted_model, "real_fast_tritopic", {"dataset_name": "test_ds"}
    )
    assert "representative_docs" in qual_df.columns
    assert "representation" in qual_df.columns
    assert qual_df["representative_docs"].dtype == pl.List(pl.String)
    assert qual_df["representation"].dtype == pl.List(pl.String)

    rep_docs_sample = qual_df["representative_docs"].to_list()[0]
    assert len(rep_docs_sample) > 0
    assert rep_docs_sample[0] in texts
