# -*- coding: utf-8 -*-
"""Tests for the dataset builders."""

import polars as pl
import pytest

from src.builders.fed import load_macro_data
from src.builders.yelp import change_lf


@pytest.fixture
def temp_raw_dir(tmp_path):
    """Creates a temporary directory for raw files."""
    d = tmp_path / "raw"
    d.mkdir()
    return d


def test_fed_load_macro_data(temp_raw_dir):
    """Tests the macro data loading and transformation logic
    (interpolation and lagging).
    """
    # Create dummy macro data with a null and a gap
    # Row 2 has a null that should be filled by rolling mean
    csv_path = temp_raw_dir / "test_macro.csv"
    df = pl.DataFrame(
        {
            "observation_date": [
                "2000-01-01",
                "2000-02-01",
                "2000-03-01",
                "2000-04-01",
                "2000-05-01",
            ],
            "VALUE": [10.0, None, 12.0, 13.0, 14.0],
        }
    )
    df.write_csv(csv_path)

    # Expected value for the null (row index 1):
    # rolling mean window 5 centered: (10 + 12 + 13) / 3 = 35 / 3 ≈ 11.666

    result = load_macro_data(temp_raw_dir, "test_macro.csv", "VALUE", "new_val")

    assert result.height == 5
    assert "new_val" in result.columns
    assert "new_val_lag" in result.columns

    # Check interpolation (the None was at index 1)
    assert result["new_val"][1] is not None
    assert round(result["new_val"][1], 2) == 11.67

    # Check lagging
    # Row 1 value should be in Row 2 lag
    assert result["new_val_lag"][1] == result["new_val"][0]
    # Row 0 lag should be backfilled (same as row 0 value)
    assert result["new_val_lag"][0] == result["new_val"][0]


def test_yelp_change_lf():
    """Tests the Yelp-specific column selection and renaming logic."""
    lf = pl.LazyFrame(
        {
            "date": ["2020-01-01"],
            "stars": [5],
            "text": ["Great!"],
            "review_count": [10],
            "average_stars": [4.5],
            "yelping_since": ["2015-01-01"],
            "state": ["CA"],
            "stars_business": [4.0],
            "review_count_business": [100],
            "other_col": ["garbage"],
        }
    )

    select_cols = [
        "date",
        "stars",
        "text",
        "review_count",
        "average_stars",
        "yelping_since",
        "state",
        "stars_business",
        "review_count_business",
    ]

    result_df = change_lf(lf, select_cols).collect()

    # Check that other_col was dropped
    assert "other_col" not in result_df.columns

    # Check renames
    assert "business_review_count" in result_df.columns
    assert "business_stars" in result_df.columns
    assert "user_average_stars" in result_df.columns
    assert "user_review_count" in result_df.columns

    # Verify values mapped correctly
    assert result_df["business_review_count"][0] == 100
    assert result_df["user_average_stars"][0] == 4.5
