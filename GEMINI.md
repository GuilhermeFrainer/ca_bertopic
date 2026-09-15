# CA-BERTopic Project

This project aims to modify BERTopic to incorporate document-level metadata into the topic modeling process using multi-view clustering (e.g., Multi-View K-Means, Multi-View Spectral, Co-regularized Spectral), multi-modal graph modeling (FastTriTopic/TriTopic), and Structural Topic Model (STM) benchmarks.

---

## Getting Started

### Prerequisites

*   **Python:** 3.12+
*   **Package Manager:** `uv`
*   **R (Optional):** R >= 4.0 with `renv` (required only for building Bag-of-Words and training R-based STM models)
*   **Local Dependencies:** `fast-tritopic` as a local path dependency (`../fast-tritopic`)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd CA-BERTopic
    ```

2.  Create a virtual environment and install dependencies using `uv`:
    ```bash
    uv venv
    uv sync
    ```
    > [!NOTE]
    > Linux- and NVIDIA GPU-specific dependencies (e.g., `cudf`, `cuml`, `cugraph`) are marked with `sys_platform == 'linux'` in `pyproject.toml`. On Windows or macOS, `uv sync` resolves CPU dependencies cleanly without modifying `pyproject.toml`.

---

## Supported Datasets & Ingestion Pipeline

The project supports five core datasets:
*   `anes`: American National Election Studies open-ended survey responses with political covariates.
*   `fed`: Federal Reserve communications linked with macroeconomic indicators and political party metadata.
*   `gadarian`: Open-ended responses regarding public health and emotion with demographic covariates.
*   `trump`: Social media posts linked with engagement and timestamp metadata.
*   `yelp` / `yelp_s10000`: Business reviews joined with business metadata and star ratings (subsampled to 10k aligned documents for parity with STM).

### Standardized Multi-Stage Data Pipeline

To eliminate confounding between models, **all text preprocessing is executed 100% in Python**. R scripts ingest preprocessed Parquet text directly, without performing any R-side stopword filtering or stemming.

1.  **Build Unified Raw/Interim Datasets:**
    ```bash
    uv run scripts/data_prep/build_datasets.py --dataset <dataset_name>
    ```

2.  **Align and Subsample (Yelp Only):**
    ```bash
    uv run scripts/data_prep/align_yelp_sample.py
    ```

3.  **Preprocess Text (Dual Representation):**
    Generates both `clean_text` (unstemmed, casing/syntax preserved for SentenceTransformers) and `clean_text_stemmed` (lowercased, NLTK stopwords removed, Snowball stemmed for classical BoW/STM models), enforcing strict row alignment:
    ```bash
    uv run scripts/data_prep/preprocess_datasets.py --dataset <dataset_name>
    ```

4.  **Generate Dual Embeddings:**
    Computes dense embeddings for both text columns using `all-MiniLM-L6-v2`:
    ```bash
    uv run scripts/data_prep/generate_embeddings.py --dataset <dataset_name> --columns clean_text clean_text_stemmed
    ```

5.  **Build BoW and STM Objects (R):**
    ```bash
    # Unstemmed representation
    Rscript scripts/r_scripts/build_bow.R --dataset <dataset_name> --text_col clean_text

    # Stemmed representation
    Rscript scripts/r_scripts/build_bow.R --dataset <dataset_name> --text_col clean_text_stemmed --output_suffix _stemmed
    ```

6.  **Automated Representation Pipeline (Windows PowerShell):**
    To regenerate all datasets and representations in a single pass:
    ```powershell
    .\scripts\pipelines\local_windows\build_representations.ps1
    ```

---

## Running Experiments

Experiment configurations are defined by `.yaml` files in the `experiments/` directory and executed via `scripts/experiments/run_experiment.py`.

Active production experiments are organized by dataset:
- `experiments/<dataset>/`: Standard unstemmed runs (`<dataset>_standard_*.yaml`).
- `experiments/<dataset>_stemmed/`: Stemmed text runs (`<dataset>_stemmed_standard_*.yaml`).
- `experiments/archive/<dataset>/`: Archived optimization, ablation, and exploratory runs.

### Command-Line Execution

To run an experiment:
```bash
uv run python scripts/experiments/run_experiment.py --exp <dataset>/<config_name>
```
*(You may omit the directory prefix and `.yaml` extension if the configuration name is unique, e.g., `--exp trump_standard_baseline`)*.

#### Examples:
```bash
# Run Trump standard baseline experiment
uv run python scripts/experiments/run_experiment.py --exp trump/trump_standard_baseline

# Run FED Multi-View K-Means experiment
uv run python scripts/experiments/run_experiment.py --exp fed/fed_standard_mv_k_means

# Fast dry-run with a single seed and subsample
uv run python scripts/experiments/run_experiment.py --exp trump/trump_standard_baseline --sample 500 --single-seed
```

### Representation Stop Words Removal (`--remove-rep-stopwords`)

By default, BERTopic's c-TF-IDF representation layer removes English stop words via `CountVectorizer(stop_words="english")` (`--remove-rep-stopwords`). This ensures extracted topic keywords are informative without altering natural sentence structure in the transformer embeddings. To retain stop words in topic representations, pass `--keep-rep-stopwords`.

### Running STM Baseline Experiments

To run Structural Topic Model baselines via R:
```bash
# Standard unstemmed STM
uv run python scripts/experiments/run_stm.py --dataset fed

# Stemmed STM
uv run python scripts/experiments/run_stm.py --dataset fed --stemmed
```

### Running Hyperparameter Optimization

Optimization configurations specify parameter search spaces as lists (e.g., `n_clusters: [30, 50, 80]`) and are located in `experiments/archive/<dataset>/`:
```bash
uv run python scripts/experiments/run_optimizer.py --exp archive/yelp/yelp_opt_mv_spectral
```

---

## Results Management & Analysis

### Result Type Separation

Experiments produce results across three isolated preprocessing regimes:
1.  `standard`: Models operating on unstemmed `clean_text`.
2.  `stemmed`: Models operating on stemmed, stopword-removed `clean_text_stemmed`.
3.  `no_stopword_removal`: Legacy baseline models evaluated before stopword filtering.

Analysis utilities support the `--result-type` parameter (`standard`, `stemmed`, `no_stopword_removal`, or `all`) to prevent metrics from being cross-contaminated.

### Merging Results & Single-Pass Archival

Individual experiment runs generate raw CSVs in `results/` and topic JSON files in `output/`. Consolidate these into unified dataset files using:
```bash
uv run python scripts/analysis/merge_results.py
```
This script performs single-pass consolidation:
- Merges latest runs into `results/<dataset>_<type>_merged.csv` and `output/<dataset>_<type>_merged.json`.
- Pools all contributing and superseded raw run files into timestamped ZIP archives (`results/archive/` and `output/archive/`).
- Deletes unmerged raw files from disk, avoiding orphaned artifacts.

### Scraping Best Models & Generating LaTeX Tables

To extract top-performing models per metric and generate publication-ready LaTeX tables:
```bash
# Display best models for a dataset
uv run python scripts/analysis/find_best_models.py --dataset fed --result-type standard

# Export formatted LaTeX table
uv run python scripts/analysis/find_best_models.py --dataset fed --result-type standard --latex tables/fed_table.tex
```

### Noise Coverage Calculation

To evaluate and format HDBSCAN noise coverage into LaTeX:
```bash
uv run python scripts/analysis/calculate_noise_coverage.py --dataset fed --result-type standard --latex
```

### Automated Results Pipeline (Windows PowerShell)

To scrape, merge, and export all LaTeX tables and figures across all result types to the dissertation output directory:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/pipelines/local_windows/get_results.ps1 -Release
```

---

## Visualizing Results

The project includes an interactive Streamlit dashboard to explore and compare experiment metrics:
```bash
uv run streamlit run scripts/dashboard.py
```

The dashboard enables:
*   Filtering by dataset, model type, date, and preprocessing regime (`standard`, `stemmed`, `no_stopword_removal`).
*   Direct comparison across coherence (`c_v`, `u_mass`), diversity (`irbo`), and outlier metrics.
*   Interactive scatter plots and automated highlighting of best models.

---

## Running Tests

The test suite contains 460+ unit and integration tests covering builders, models, config inheritance, and merge pipelines:
```bash
uv run pytest
```

---

## Project Structure

```
├── data/                      # Raw, interim, and processed datasets (.parquet, .rds)
├── docs/                      # Architectural guides and technical documentation
│   ├── archived_experiments_summary.md
│   ├── decoupled_multiview_distance_metrics.md
│   ├── fast_tritopic_implementation_plan.md
│   ├── merge_results_lifecycle.md
│   ├── preprocessing.md
│   ├── project_structure.md
│   ├── representation_stopwords.md
│   └── results_separation.md
├── experiments/               # Experiment configuration files
│   ├── anes/                  # Active standard ANES configs
│   ├── fed/                   # Active standard FED configs
│   ├── gadarian/              # Active standard Gadarian configs
│   ├── trump/                 # Active standard Trump configs
│   ├── yelp/                  # Active standard Yelp configs
│   ├── *_stemmed/             # Active standard stemmed configs
│   ├── datasets/              # Base dataset definitions (inherited via 'extends')
│   └── archive/               # Archived optimization and exploratory configs
├── models/                    # Serialized model artifacts
├── notebooks/                 # Jupyter & Marimo exploratory notebooks
├── output/                    # Qualitative topic representations (.json)
├── results/                   # Metric evaluation results (.csv) and archives (.zip)
├── scripts/                   # Utility scripts and execution pipelines
│   ├── data_prep/             # Ingestion, preprocessing, embeddings, sampling
│   ├── experiments/           # Experiment runners, optimizer, STM coordinator
│   ├── analysis/              # Results merge, best models scraper, noise coverage
│   ├── pipelines/             # Local Windows (.ps1), Linux (.sh), and SLURM runners
│   ├── r_scripts/             # R scripts for Bag-of-Words and STM training
│   ├── temp/                  # Temporary test sandboxes
│   └── dashboard.py           # Streamlit results dashboard
├── src/                       # Core Python library
│   ├── builders/              # Dataset-specific ingestion builders
│   ├── append_umap.py         # AppendUMAP dimension reduction wrapper
│   ├── data.py                # Dataset loading and splitting
│   ├── decoupled_kmeans.py    # Decoupled Multi-View K-Means implementations
│   ├── decoupled_spectral.py  # Decoupled Multi-View Spectral clustering
│   ├── embeddings.py          # SentenceTransformers embedding generation
│   ├── evaluation.py          # Metric calculations (c_v, u_mass, irbo)
│   ├── experiment_queue.py    # Experiment queue orchestration
│   ├── experiment_tracker.py  # Experiment run tracking and persistence
│   ├── logger_config.py       # Centralized logging configuration
│   ├── make_table.py          # Great Tables & LaTeX table generators
│   ├── models.py              # BERTopic, Multi-View, and TriTopic integrations
│   ├── mvc_wrapper.py         # Multi-View Clustering wrappers
│   ├── optimizer.py           # Hyperparameter optimization engine
│   ├── processing.py          # Standardized text cleaning and stemming
│   ├── results_analysis.py    # Results parsing and model type classification
│   ├── training.py            # Training routines
│   ├── utils.py               # Config loading and helper utilities
│   ├── verification.py        # Config and dataset verification tools
│   └── visualization.py       # Plotly chart generators
├── tables/                    # Generated LaTeX and Great Tables outputs
├── tests/                     # Comprehensive pytest test suite
└── pyproject.toml             # Project dependencies and configuration
```

---

## Core Technologies

*   **BERTopic:** Modular topic modeling framework.
*   **mvlearn:** Multi-view learning algorithms (Multi-View K-Means, Multi-View Spectral, Co-regularized Spectral).
*   **FastTriTopic / TriTopic:** Multi-modal graph topic modeling integrating text embeddings and document metadata via sparse graph laplacians with vectorized coordinate construction.
*   **Structural Topic Model (STM) / R (`quanteda`, `stm`):** Semi-parametric topic modeling incorporating document covariates.
*   **SentenceTransformers:** Contextual text representation models (default: `all-MiniLM-L6-v2`).
*   **OCTIS & gensim:** Coherence evaluation metrics (`c_v`, `u_mass`).
*   **Polars & PyArrow:** High-performance tabular data manipulation and Parquet storage.
*   **Great Tables & Plotly:** Publication-grade LaTeX/HTML tables and interactive figures.
*   **uv:** Fast Python packaging and project management.

---

## Gemini CLI Architectural Mandates & Guidelines

1.  **Test-Driven Development & Verification:**
    - Always run the full test suite (`uv run pytest`) after implementing new features, fixing bugs, or modifying configurations. Ensure all 460+ tests pass before marking tasks complete.
2.  **Linting and Formatting:**
    - Format and lint code changes using Ruff (`uvx ruff check . --fix` and `uvx ruff format .`).
3.  **Portable Documentation Links:**
    - Use relative links in markdown documentation files (e.g., `[project_structure.md](project_structure.md)` or `[src/data.py](../src/data.py)`). Never use absolute paths or `file:///` URIs in repository markdown files.
4.  **Python-Only Preprocessing Parity:**
    - All text normalization, stopword filtering, and stemming must strictly occur in Python (`src/processing.py`). R scripts must only ingest pre-processed strings from Parquet to ensure zero preprocessing discrepancies between Python models and R baselines.
5.  **Strict Row Alignment:**
    - Documents must only be retained if non-empty in **both** `clean_text` and `clean_text_stemmed`. Never filter one column independently.
6.  **Representation Stopwords Filter:**
    - Filter stop words at the c-TF-IDF topic representation layer (`CountVectorizer(stop_words="english")`) rather than stripping words from embedding inputs, preserving contextual language structure.
7.  **Result Type Isolation:**
    - Never merge or compare `standard`, `stemmed`, and `no_stopword_removal` runs into a single unsegregated analysis pool. Always respect the `--result-type` filter.
8.  **Single-Pass Archival:**
    - Ensure results merging utilities track both latest and superseded raw run files, packing them into timestamped archives and cleaning the working directory in a single pass.
9.  **Coding Style:**
    - Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
