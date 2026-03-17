# -*- coding: utf-8 -*-
"""Tests for ID uniqueness in dataset preprocessing."""

import pytest
import polars as pl
from src.processing import process_dataset, BATCH_SIZE

def test_index_uniqueness_across_batches(tmp_path):
    """
    Tests that the 'index' column remains unique even when the dataset 
    is processed in multiple batches and rows are chunked.
    """
    # Create a dataset larger than BATCH_SIZE to force multiple batches
    # BATCH_SIZE is 10000 in src/processing.py
    num_rows = BATCH_SIZE + 500
    
    # Create some rows that will definitely be chunked
    long_text = ". ".join(["This is a long sentence that should be split eventually"] * 20)
    
    df = pl.DataFrame({
        "text": [long_text if i % 100 == 0 else f"Short text {i}" for i in range(num_rows)],
        "user_review_count": [i for i in range(num_rows)],
        "business_review_count": [i * 2 for i in range(num_rows)],
        "state": ["CA"] * num_rows,
    })
    
    input_path = tmp_path / "large_yelp.parquet"
    output_path = tmp_path / "large_yelp_processed.parquet"
    df.write_parquet(input_path)
    
    # Process the dataset with a small max_tokens to ensure chunking happens
    processed_df = process_dataset(
        dataset_name="yelp",
        input_path=str(input_path),
        output_path=str(output_path),
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=50, # Small limit to force chunking of the long_text rows
    )
    
    # 1. Check that 'index' column exists (this is the unique row identifier)
    assert "index" in processed_df.columns
    
    # 2. Check that 'index' values are unique
    assert processed_df["index"].is_unique().all()
    assert processed_df["index"].n_unique() == processed_df.height
    
    # 3. Verify that the number of rows increased due to chunking
    assert processed_df.height > num_rows

def test_index_uniqueness_with_existing_columns(tmp_path):
    """
    Tests that if an 'index' column already exists in the input,
    it is correctly replaced to ensure global uniqueness after processing.
    """
    df = pl.DataFrame({
        "index": [999, 999], # Duplicate index in input
        "text": ["Short text 1", "Short text 2"],
        "user_review_count": [1, 2],
        "business_review_count": [10, 20],
        "state": ["CA", "NY"],
    })
    
    input_path = tmp_path / "existing_index.parquet"
    output_path = tmp_path / "existing_index_processed.parquet"
    df.write_parquet(input_path)
    
    processed_df = process_dataset(
        dataset_name="yelp",
        input_path=str(input_path),
        output_path=str(output_path),
        max_tokens=100,
    )
    
    # The old 'index' [999, 999] should be gone and replaced by unique indices [0, 1]
    assert "index" in processed_df.columns
    assert processed_df["index"].to_list() == [0, 1]
    assert processed_df["index"].is_unique().all()
