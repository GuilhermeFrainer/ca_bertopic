import polars as pl
import pytest
from pathlib import Path
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results_analysis import find_best_models, extract_model_type

def test_extract_model_type():
    assert extract_model_type("baseline_1") == "baseline"
    assert extract_model_type("mv_co_reg_spectral_1") == "mv_co_reg_spectral"
    assert extract_model_type("pca_mv_spectral_info0_10") == "pca_mv_spectral_info0"
    assert extract_model_type("vanilla") == "vanilla"

def test_find_best_models():
    data = {
        "model_name": ["baseline_1", "baseline_2", "mv_1", "mv_2"],
        "dataset_name": ["fed", "fed", "fed", "fed"],
        "c_v": [0.5, 0.7, 0.6, 0.8],
        "u_mass": [-1.5, -1.2, -1.4, -1.1],
        "n_topics": [10, 20, 10, 20]
    }
    df = pl.DataFrame(data)
    
    results = find_best_models(df, "fed")
    
    assert "c_v" in results
    assert "u_mass" in results
    
    cv_results = results["c_v"]
    assert cv_results.height == 2
    
    assert cv_results.filter(pl.col("model_type") == "baseline")["max_value"][0] == 0.7
    assert cv_results.filter(pl.col("model_type") == "baseline")["best_model_name"][0] == "baseline_2"
    
    assert cv_results.filter(pl.col("model_type") == "mv")["max_value"][0] == 0.8
    assert cv_results.filter(pl.col("model_type") == "mv")["best_model_name"][0] == "mv_2"

def test_find_best_models_filtering():
    data = {
        "model_name": ["baseline_1", "baseline_2"],
        "dataset_name": ["fed", "yelp"],
        "c_v": [0.5, 0.9]
    }
    df = pl.DataFrame(data)
    
    results_fed = find_best_models(df, "fed")
    assert results_fed["c_v"].filter(pl.col("model_type") == "baseline")["max_value"][0] == 0.5
    
    results_yelp = find_best_models(df, "yelp")
    assert results_yelp["c_v"].filter(pl.col("model_type") == "baseline")["max_value"][0] == 0.9

def test_find_best_models_exclusion():
    data = {
        "model_name": ["pca_mv_1", "umap_mv_1", "k_means_1", "spectral_1"],
        "dataset_name": ["fed", "fed", "fed", "fed"],
        "clustering_algo": ["mv", "mv", "k_means", "spectral"],
        "dim_red_algo": ["pca", "umap", "umap", "umap"],
        "c_v": [0.5, 0.6, 0.7, 0.8]
    }
    df = pl.DataFrame(data)
    
    # Exclude PCA
    res_no_pca = find_best_models(df, "fed", exclude_dim_red=["pca"])
    # Should only have umap_mv_1, k_means_1, spectral_1
    all_best_names = []
    for m_df in res_no_pca.values():
        all_best_names.extend(m_df["best_model_name"].to_list())
    assert "pca_mv_1" not in all_best_names
    
    # Exclude K-Means
    res_no_kmeans = find_best_models(df, "fed", exclude_clustering=["k_means"])
    all_best_names = []
    for m_df in res_no_kmeans.values():
        all_best_names.extend(m_df["best_model_name"].to_list())
    assert "k_means_1" not in all_best_names
