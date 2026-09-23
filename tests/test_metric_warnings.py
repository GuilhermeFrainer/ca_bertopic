"""Non-finite evaluation scores must remain visible without aborting a run."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest

from src import training


@pytest.fixture(autouse=True)
def capture_pipeline_logs(monkeypatch):
    """Other tests configure the pipeline logger without propagation."""
    monkeypatch.setattr(logging.getLogger("pipeline"), "propagate", True)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metrics_warn_and_continue(monkeypatch, caplog, score):
    model = Mock()
    model.fit_transform.return_value = ([0, 0], None)
    model.get_topics.return_value = {0: []}
    model.vectorizer_model.build_analyzer.return_value = str.split
    monkeypatch.setattr(
        training.evaluation,
        "bertopic_output_to_octis",
        lambda model: {"topics": [["assistant", "general counselmr"]]},
    )
    monkeypatch.setattr(training.evaluation, "compute_coherence", lambda **kw: score)
    diversity = Mock(return_value=0.75)
    monkeypatch.setattr(training.evaluation, "compute_diversity", diversity)

    with caplog.at_level(logging.WARNING, logger="pipeline"):
        metrics, fitted = training.train_and_evaluate(
            model,
            "test_model",
            ["assistant general", "assistant counselmr"],
            np.zeros((2, 3)),
            {
                "experiment": {
                    "coherence_metrics": ["u_mass"],
                    "diversity_metrics": ["irbo"],
                }
            },
        )

    assert fitted is model
    assert not np.isfinite(metrics["u_mass"])
    assert metrics["irbo"] == 0.75
    diversity.assert_called_once()
    assert "[test_model] Evaluation metric u_mass" in caplog.text
    assert "Keeping the score and continuing" in caplog.text
    assert "1 keywords absent" in caplog.text
    assert "1 topics with fewer than two distinct" in caplog.text


def test_nonfinite_diversity_warns(monkeypatch, caplog):
    model = Mock()
    model.fit_transform.return_value = ([0], None)
    model.get_topics.return_value = {0: []}
    model.vectorizer_model.build_analyzer.return_value = str.split
    monkeypatch.setattr(
        training.evaluation, "bertopic_output_to_octis", lambda model: {}
    )
    monkeypatch.setattr(
        training.evaluation, "compute_diversity", lambda *args, **kw: float("nan")
    )
    with caplog.at_level(logging.WARNING, logger="pipeline"):
        training.train_and_evaluate(
            model,
            "diversity_model",
            ["document"],
            np.zeros((1, 3)),
            {"experiment": {"coherence_metrics": [], "diversity_metrics": ["irbo"]}},
        )
    assert "Evaluation metric irbo" in caplog.text
    assert "Coherence diagnostics" not in caplog.text
