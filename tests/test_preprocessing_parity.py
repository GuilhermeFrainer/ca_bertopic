# -*- coding: utf-8 -*-
"""Tests for text preprocessing logic and parity between Python and R representations."""

import tempfile
from pathlib import Path

import nltk
import polars as pl

from src.processing import process_dataset, stem_and_remove_stopwords


def test_stem_and_remove_stopwords():
    """Verify that stem_and_remove_stopwords lowercases, strips punctuation, removes stopwords, and stems."""
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    stop_words = set(nltk.corpus.stopwords.words("english"))

    raw_text = "The quick brown foxes were running fast through 100 cities!"
    # "the", "were", "through" are stopwords
    # "100" or numbers were already stripped or ignored
    processed = stem_and_remove_stopwords(raw_text, stemmer, stop_words)
    tokens = processed.split()

    assert "the" not in tokens
    assert "were" not in tokens
    assert "through" not in tokens
    assert "fox" in tokens or "foxes" in tokens
    assert "run" in tokens
    assert "citi" in tokens


def test_dual_preprocessing_alignment():
    """Verify clean_text and clean_text_stemmed row alignment and empty handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "test_input.csv"
        output_parquet = Path(tmpdir) / "test_output.parquet"

        df = pl.DataFrame(
            {
                "text": [
                    "This is a valid document about political economy.",
                    "The and in on at",  # Only stopwords!
                    "http://example.com 12345 !!!",  # Only noise!
                    "Another valid document for testing topic modeling.",
                ]
            }
        )
        df.write_csv(input_csv)

        result_df = process_dataset(
            dataset_name="gadarian",
            input_path=str(input_csv),
            output_path=str(output_parquet),
            stem=True,
        )

        # "The and in on at" becomes empty in clean_text_stemmed
        # "http://example.com 12345 !!!" becomes empty in clean_text
        # Both should be filtered out to keep exact alignment between clean_text and clean_text_stemmed
        assert len(result_df) == 2
        assert "clean_text" in result_df.columns
        assert "clean_text_stemmed" in result_df.columns
        assert (result_df["clean_text"].str.strip_chars() != "").all()
        assert (result_df["clean_text_stemmed"].str.strip_chars() != "").all()
