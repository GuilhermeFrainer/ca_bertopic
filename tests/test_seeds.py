from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

from src.optimizer import Optimizer
from src.utils import get_random_state


def test_get_random_state_list():
    # Test single int
    assert get_random_state(123) == 123

    # Test random
    res = get_random_state("random")
    assert isinstance(res, int)
    assert 0 <= res <= 100_000

    # Test list of ints
    assert get_random_state([123, 456]) == [123, 456]

    # Test list of "random"
    res_list = get_random_state(["random", 123, "random"])
    assert len(res_list) == 3
    assert isinstance(res_list[0], int)
    assert res_list[1] == 123
    assert isinstance(res_list[2], int)


def test_optimizer_flattening():
    # Mocking texts, embeddings, scaled_metadata
    texts = ["hello", "world"]
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaled_metadata = np.array([[0.5], [0.5]])

    # Mock Optimizer class with standard mock data
    model_config = {
        "id": "mock_model",
        "clustering": {"type": "kmeans", "params": {"n_clusters": [2, 3]}},
        "dimensionality_reduction": {"type": "umap", "params": {}},
    }

    experiment_config = {
        "experiment": {
            "name": "test_exp",
            "random_state": [36201624, 62613654, 57116123],
            "dataset_path": "data/processed/test_embeddings.parquet",
            "coherence_metrics": ["u_mass"],
            "diversity_metrics": ["topic_diversity"],
        }
    }

    # Setup the Optimizer
    optimizer = Optimizer(
        texts=texts,
        embeddings=embeddings,
        scaled_metadata=scaled_metadata,
        model_config=model_config,
        experiment_config=experiment_config,
        experiment_id="test_exp",
        random_state=[36201624, 62613654, 57116123],
        file_timestamp="20260630-100000",
    )

    # Patch create_bertopic_instance, train_and_evaluate, and extract_qualitative_data
    with (
        patch("src.models.create_bertopic_instance") as mock_create,
        patch("src.training.train_and_evaluate") as mock_train,
        patch("src.utils.extract_qualitative_data") as mock_qual,
    ):
        # Setup mock returns
        mock_create.return_value = MagicMock()
        mock_train.side_effect = lambda topic_model, model_id, *args, **kwargs: (
            {
                "model_name": model_id,
                "duration_seconds": 1.2,
                "n_topics": 3,
                "outliers": 0,
                "u_mass": -0.5,
                "topic_diversity": 0.8,
            },
            MagicMock(),  # trained model
        )
        mock_qual.return_value = pl.DataFrame()

        # 1. Run all combinations without target_index
        optimizer.results = []
        optimizer.run(start_index=0, target_index=None)

        # Hyperparameter combinations generated:
        # n_clusters = 2
        # n_clusters = 3
        # Total combinations: 2
        # Total seeds: 3
        # Total runs: 2 * 3 = 6
        assert mock_train.call_count == 6

        # Verify the seeds passed to models.create_bertopic_instance
        assert len(optimizer.results) == 6
        seeds_run = [r["random_state"] for r in optimizer.results]
        expected_seeds = [36201624, 62613654, 57116123, 36201624, 62613654, 57116123]
        assert seeds_run == expected_seeds

        # 2. Run target_index (index 2: combo_idx 0, seed 3)
        mock_train.reset_mock()
        optimizer.results = []
        optimizer.run(start_index=0, target_index=2)

        assert mock_train.call_count == 1
        assert len(optimizer.results) == 1
        assert optimizer.results[0]["random_state"] == 57116123
        assert optimizer.results[0]["model_name"] == "mock_model_1_seed57116123"
