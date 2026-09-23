"""Keep experiment keyword settings consistent across both TriTopic variants."""

import copy
from pathlib import Path

import pytest
import yaml

from src.models import create_topic_model_instance
from src.optimizer import generate_hyperparameter_combinations

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CONFIGS = sorted(
    path for path in EXPERIMENTS.glob("*/*tritopic.yaml") if "archive" not in path.parts
)


@pytest.mark.parametrize(
    "path", CONFIGS, ids=lambda path: str(path.relative_to(EXPERIMENTS))
)
def test_experiment_unigrams_reach_model(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
    original = copy.deepcopy(config)
    assert config["params"]["keyword_ngram_range"] == [1, 1]
    combinations = generate_hyperparameter_combinations(config)
    assert len(combinations) == len(config["params"]["n_topics"])
    for resolved, varied in combinations:
        assert resolved["params"]["keyword_ngram_range"] == [1, 1]
        assert "keyword_ngram_range" not in varied
        model = create_topic_model_instance(resolved, None, random_state=42)
        assert model._keyword_extractor.ngram_range == (1, 1)
    assert config == original


@pytest.mark.parametrize("model_type", ["tritopic", "fast_tritopic"])
@pytest.mark.parametrize("ngram_range", [[1, 1], [2, 2]])
def test_extracted_keywords_obey_range_after_reset(model_type, ngram_range):
    config = {"type": model_type, "params": {"keyword_ngram_range": ngram_range}}
    model = create_topic_model_instance(config, None, random_state=42)
    extractor = model._keyword_extractor
    documents = [
        "apple banana orchard",
        "apple banana fruit",
        "car train road",
        "car train travel",
    ]
    for _ in range(2):
        # TriTopic resets the extractor at fit time; the setting must survive.
        extractor.reset()
        words, _ = extractor.extract(documents[:2], all_docs=documents)
        assert words
        assert all(len(word.split()) == ngram_range[0] for word in words)
        assert extractor._vectorizer.ngram_range == tuple(ngram_range)


@pytest.mark.parametrize("model_type", ["tritopic", "fast_tritopic"])
def test_omitted_range_preserves_upstream_default(model_type):
    model = create_topic_model_instance({"type": model_type}, None, random_state=42)
    assert model._keyword_extractor.ngram_range == (1, 2)


@pytest.mark.parametrize("model_type", ["tritopic", "fast_tritopic"])
@pytest.mark.parametrize(
    "value", [None, 1, [], [1], [0, 1], [2, 1], [1, 1.5], [True, 1]]
)
def test_invalid_range_rejected(model_type, value):
    with pytest.raises(ValueError, match="keyword_ngram_range"):
        create_topic_model_instance(
            {"type": model_type, "params": {"keyword_ngram_range": value}},
            None,
            random_state=42,
        )


@pytest.mark.parametrize("component", [None, "tritopic", "fast_tritopic"])
def test_nested_ranges_can_be_searched(component):
    params = {"keyword_ngram_range": [[1, 1], [1, 2]], "n_topics": [10, 20]}
    config = (
        {"params": params} if component is None else {component: {"params": params}}
    )
    combinations = generate_hyperparameter_combinations(config)
    assert len(combinations) == 4
    ranges = []
    for resolved, _ in combinations:
        owner = resolved if component is None else resolved[component]
        ranges.append(tuple(owner["params"]["keyword_ngram_range"]))
    assert set(ranges) == {(1, 1), (1, 2)}
