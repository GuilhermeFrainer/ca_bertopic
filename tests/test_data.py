import numpy as np
import polars as pl
import pytest

from src.data import process_metadata, sample_from_lf


# Most basic test case
def test_no_cols_produces_empty_array():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )
    covariates_config = {}

    result = process_metadata(df, covariates_config)

    assert isinstance(result, pl.DataFrame)
    assert result.is_empty()
    assert result.shape == (0, 0)


def test_numerical_scaling():
    df = pl.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "num2": [10, 20, 30, 40, 50],
        }
    )
    covariates_config = {"numerical": ["num1", "num2"]}

    result = process_metadata(df, covariates_config)

    assert isinstance(result, pl.DataFrame)
    assert result.shape == (5, 2)
    assert result.columns == ["num1", "num2"]
    arr = result.to_numpy()
    # Min-max scaling should result in values between 0 and 1
    assert np.all((arr >= 0) & (arr <= 1))

    # Check if scaling is correct
    expected = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.75, 0.75],
            [1.0, 1.0],
        ]
    )
    assert np.allclose(arr, expected)


def test_categorical_encoding():
    df = pl.DataFrame(
        {
            "cat1": ["a", "b", "a", "c"],
            "cat2": ["x", "y", "y", "x"],
        }
    )
    covariates_config = {"categorical": ["cat1", "cat2"]}

    result = process_metadata(df, covariates_config)

    assert isinstance(result, pl.DataFrame)
    # "a", "b", "c" -> 3 columns, "x", "y" -> 2 columns. Total 5
    assert result.shape == (4, 5)
    assert result.columns == [
        "cat1_a",
        "cat1_b",
        "cat1_c",
        "cat2_x",
        "cat2_y",
    ]
    arr = result.to_numpy()
    # One-hot encoding should result in 0s and 1s
    assert np.all(np.isin(arr, [0, 1]))

    # Expected: cat1_a, cat1_b, cat1_c, cat2_x, cat2_y
    expected = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ]
    )
    assert np.allclose(arr, expected)


def test_binary_casting():
    df = pl.DataFrame(
        {
            "bin1": [True, False, True, True],
            "bin2": [0, 1, 1, 0],
        }
    )
    covariates_config = {"binary": ["bin1", "bin2"]}

    result = process_metadata(df, covariates_config)

    assert isinstance(result, pl.DataFrame)
    assert result.shape == (4, 2)
    assert result.columns == ["bin1", "bin2"]
    for col in result.columns:
        assert result[col].dtype == pl.Float64

    arr = result.to_numpy()
    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    )
    assert np.allclose(arr, expected)


def test_mixed_types():
    df = pl.DataFrame(
        {
            "num": [10, 20],
            "cat": ["a", "b"],
            "bin": [True, False],
        }
    )
    covariates_config = {
        "numerical": ["num"],
        "categorical": ["cat"],
        "binary": ["bin"],
    }

    result = process_metadata(df, covariates_config)

    assert isinstance(result, pl.DataFrame)
    # num (1) + cat_a, cat_b (2) + bin (1) = 4 columns
    assert result.shape == (2, 4)
    assert result.columns == ["num", "cat_a", "cat_b", "bin"]

    arr = result.to_numpy()
    expected = np.array(
        [
            # num, cat_a, cat_b, bin
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    assert np.allclose(arr, expected)


def test_missing_column_raises_error():
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match=r"Missing numerical columns: \['b'\]"):
        process_metadata(df, {"numerical": ["b"]})

    with pytest.raises(ValueError, match=r"Missing categorical columns: \['c'\]"):
        process_metadata(df, {"categorical": ["c"]})

    with pytest.raises(ValueError, match=r"Missing binary columns: \['d'\]"):
        process_metadata(df, {"binary": ["d"]})


def test_numerical_zero_division():
    df = pl.DataFrame({"num": [5, 5, 5, 5]})
    covariates_config = {"numerical": ["num"]}

    result = process_metadata(df, covariates_config)

    assert isinstance(result, pl.DataFrame)
    assert result.shape == (4, 1)
    arr = result.to_numpy()
    # Should not produce NaNs and be all zeros
    assert not np.isnan(arr).any()
    assert np.all(arr == 0)


def test_legacy_config():
    df = pl.DataFrame({"num1": [1, 2, 3], "num2": [10, 20, 30]})
    # Old-style list config
    covariates_config = ["num1", "num2"]

    result = process_metadata(df, covariates_config)

    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 2)
    arr = result.to_numpy()
    assert np.all((arr >= 0) & (arr <= 1))


# --- Tests for sample_from_lf ---


@pytest.fixture
def sample_lf():
    """Provides a sample LazyFrame for testing."""
    n_rows = 100
    return pl.DataFrame(
        {
            # we need the index column for the sampling function to work
            "index": np.arange(n_rows),
            "values": np.arange(n_rows),
        }
    ).lazy()


def test_sample_returns_correct_size(sample_lf):
    n = 20
    sampled_lf = sample_from_lf(sample_lf, n=n, seed=42)

    assert isinstance(sampled_lf, pl.LazyFrame)

    sampled_df = sampled_lf.collect()
    assert len(sampled_df) == n


def test_sample_is_reproducible_with_seed(sample_lf):
    sample1 = sample_from_lf(sample_lf, n=10, seed=42).collect()
    sample2 = sample_from_lf(sample_lf, n=10, seed=42).collect()
    sample3 = sample_from_lf(sample_lf, n=10, seed=1337).collect()

    # Same seed should produce the same result
    assert sample1.equals(sample2)
    # Different seed should produce a different result
    assert not sample1.equals(sample3)


def test_sample_without_replacement_raises_error(sample_lf):
    n_larger_than_df = 200  # sample_lf has 100 rows

    with pytest.raises(ValueError):
        sample_from_lf(sample_lf, n=n_larger_than_df, replace=False, seed=42)


def test_sample_with_replacement(sample_lf):
    n_larger_than_df = 200

    # This should not raise an error
    sampled_lf = sample_from_lf(sample_lf, n=n_larger_than_df, replace=True, seed=42)
    sampled_df = sampled_lf.collect()

    assert len(sampled_df) == n_larger_than_df
    # Check if there are duplicates, which are guaranteed
    assert sampled_df["index"].is_duplicated().any()
