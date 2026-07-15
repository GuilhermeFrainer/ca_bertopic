# Project Script Structure

This document outlines the organization of scripts and execution pipelines in the CA-BERTopic project.

## Directory Overview

To maintain a clean and scalable workspace, all scripts, R files, test scripts, and execution pipelines are consolidated under the [scripts/](../scripts) directory. 

The top-level `batch/` directory has been removed, and its contents have been reorganized into subdirectories under `scripts/`.

```
├── scripts/
│   ├── data_prep/              # Dataset building, preprocessing, and feature prep
│   │   ├── align_yelp_sample.py
│   │   ├── build_datasets.py
│   │   ├── generate_embeddings.py
│   │   ├── preprocess_datasets.py
│   │   └── summarize_datasets.py
│   │
│   ├── experiments/            # Core model training/execution entry points
│   │   ├── run_experiment.py
│   │   ├── run_optimizer.py
│   │   └── run_stm.py
│   │
│   ├── analysis/               # Results evaluation and verification utilities
│   │   ├── calculate_total_time.py
│   │   ├── count_models.py
│   │   ├── find_best_models.py
│   │   └── merge_results.py
│   │
│   ├── r_scripts/              # R-specific execution scripts
│   │   ├── build_bow.R
│   │   └── train_stm.R
│   │
│   ├── pipelines/              # Orchestrators and batch runner scripts
│   │   ├── local_windows/      # Local Windows batch (.bat) and PowerShell (.ps1) [IGNORED]
│   │   ├── local_unix/         # Local Linux/macOS shell scripts (.sh) [IGNORED]
│   │   └── slurm/              # Cluster/Slurm cluster scripts (.sh) [TRACKED]
│   │
│   ├── dashboard.py            # Streamlit visual results dashboard (entrypoint)
│   │
│   └── temp/                   # Temporary testing and sandbox scripts [TRACKED]
│       ├── hello_world.R
│       └── temp_check_counts.py
```

---

## Detailed Directory Breakdown

### 1. Data Preparation ([scripts/data_prep/](../scripts/data_prep))
This directory contains scripts for data ingestion, cleaning, feature engineering, and embedding generation.
- **[build_datasets.py](../scripts/data_prep/build_datasets.py)**: Unifies multiple raw data sources or converts format (e.g., Yelp NDJSON to Parquet).
- **[preprocess_datasets.py](../scripts/data_prep/preprocess_datasets.py)**: Performs text preprocessing, cleaning, and optional stemming.
- **[generate_embeddings.py](../scripts/data_prep/generate_embeddings.py)**: Runs SentenceTransformers to compute document embeddings.
- **[align_yelp_sample.py](../scripts/data_prep/align_yelp_sample.py)**: Subsamples the Yelp dataset to 10k documents while keeping exact alignment between BERTopic (chunked) and STM (un-chunked) document IDs.
- **[summarize_datasets.py](../scripts/data_prep/summarize_datasets.py)**: Utility to output statistics (token counts, document numbers) about the processed datasets.

### 2. Experiments ([scripts/experiments/](../scripts/experiments))
Contains primary entry points to launch model training and evaluations.
- **[run_experiment.py](../scripts/experiments/run_experiment.py)**: Runs a single experiment comparing multiple model configurations defined by YAML files in the `experiments/` directory.
- **[run_optimizer.py](../scripts/experiments/run_optimizer.py)**: Handles hyperparameter tuning and optimization.
- **[run_stm.py](../scripts/experiments/run_stm.py)**: A Python wrapper to coordinate R-based Structural Topic Model training.

### 3. Analysis & Evaluation ([scripts/analysis/](../scripts/analysis))
Scripts for processing, evaluating, and compiling results.
- **[find_best_models.py](../scripts/analysis/find_best_models.py)**: Scrapes results to identify the best model configurations per metric and generates plots/LaTeX tables.
- **[merge_results.py](../scripts/analysis/merge_results.py)**: Combines results from different runs or environments.
- **[calculate_total_time.py](../scripts/analysis/calculate_total_time.py)**: Analyzes time metrics from logs to determine model throughput and compute times.
- **[count_models.py](../scripts/analysis/count_models.py)**: Summarizes completed model files.

### 4. R Scripts ([scripts/r_scripts/](../scripts/r_scripts))
Keeps R language scripts separated from the Python codebase.
- **[build_bow.R](../scripts/r_scripts/build_bow.R)**: Generates bag-of-words (BoW) representations and saves STM-compatible RDS data objects.
- **[train_stm.R](../scripts/r_scripts/train_stm.R)**: Subroutine executing the training of Structural Topic Models in R.

### 5. Pipelines & Orchestration ([scripts/pipelines/](../scripts/pipelines))
Consolidates sequential execution and batch runners.
- **`local_windows/`**: Local Windows batch `.bat` scripts and PowerShell `.ps1` files. *Note: Ignored by git to allow local modification.*
- **`local_unix/`**: Local shell scripts to fetch results. *Note: Ignored by git to allow local modification.*
- **`slurm/`**: Production SLURM scripts to dispatch jobs to clusters (e.g., `queue_standard_experiments.sh`, `queue_trump_experiments.sh`). *Note: Tracked by git for reproducibility.*

---

## Example Usage Commands

### Dataset Prep
```bash
# Build & Preprocess
uv run scripts/data_prep/build_datasets.py --dataset fed
uv run scripts/data_prep/preprocess_datasets.py --dataset fed

# Generate Embeddings
uv run scripts/data_prep/generate_embeddings.py --dataset fed --columns clean_text
```

### Running Experiments
```bash
# Run Trump experiment
uv run python scripts/experiments/run_experiment.py --exp trump

# Run Hyperparameter optimization
uv run python scripts/experiments/run_optimizer.py --exp yelp_opt_spectral
```

### Result Analysis
```bash
# Find best models and print table
uv run python scripts/analysis/find_best_models.py --dataset fed
```

### Dashboard UI
```bash
# Run Streamlit dashboard
uv run streamlit run scripts/dashboard.py
```
