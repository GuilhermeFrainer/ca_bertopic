# Results Separation & Multi-Preprocess Reporting Guide

This document explains the issue of pre-existing result fusion across preprocessing variations, why result separation was implemented, how the analysis pipeline isolates result types, and how LaTeX output tables are generated.

---

## 1. Background & Confounding Issue

Topic modeling experiments produce result CSV files stored in the `results/` directory across three distinct preprocessing regimes:

1. **Standard (Non-Stemmed)**: Runs operating on `clean_text` (unstemmed text with casing/punctuation cleaned). Output filenames match `<dataset>_standard_*.csv`.
2. **Stemmed & Stopword-Removed**: Runs operating on `clean_text_stemmed` (lowercased, NLTK stopwords removed, Snowball stemmed). Output filenames match `<dataset>_stemmed_standard_*.csv`.
3. **No Stopword Removal**: Early baseline runs conducted prior to stopword filtering. Consolidated in `<dataset>_no_stopword_removal_merged.csv`.

### Why Results Were Fused
Previously, [`scripts/analysis/find_best_models.py`](../scripts/analysis/find_best_models.py) scanned all `*.csv` files in `results/`, filtered by dataset name (`dataset_name == "fed"`), and normalized model names by stripping `stemmed_`. As a result, standard runs, stemmed runs, and pre-stopword-removal runs were concatenated into a single DataFrame. This fused metrics across different text representations, obscuring performance differences specific to each preprocessing strategy.

---

## 2. Architecture & Solution

To report these three experimental regimes independently without mixing metrics, a **Result Type Filtering Layer** was added to the Python analysis tools and PowerShell pipeline.

### A. Python Analysis Layer (`--result-type`)

Both [`scripts/analysis/find_best_models.py`](../scripts/analysis/find_best_models.py) and [`scripts/analysis/calculate_noise_coverage.py`](../scripts/analysis/calculate_noise_coverage.py) now support the `--result-type` argument:

| Mode | Filter Criteria | Target Files |
| :--- | :--- | :--- |
| `standard` | Filenames containing `<dataset>_standard` excluding `stemmed` and `no_stopword` | Standard unstemmed runs |
| `stemmed` | Filenames containing `stemmed` | Stemmed text runs |
| `no_stopword_removal` (aliases: `with_stopwords`, `no_stopword`) | Filenames containing `no_stopword` | Pre-stopword-removal runs (stopwords retained) |
| `all` | Retains all CSV files (legacy behavior) | All runs pooled together |

### B. PowerShell Orchestration (`get_results.ps1`)

The pipeline script [`scripts/pipelines/local_windows/get_results.ps1`](../scripts/pipelines/local_windows/get_results.ps1) accepts a `-ResultType` parameter (`standard`, `stemmed`, `no_stopword_removal`, or `all`). When set to `all` (default), it iterates through each result type sequentially and exports LaTeX tables (and figures) into organized subfolders:

```
CA-BERTopic-Results/ (or Dissertation release directory)
├── tables/
│   ├── standard/
│   │   ├── model_labels.tex
│   │   ├── hdbscan_noise_coverage.tex
│   │   ├── fed_table_avg.tex
│   │   └── ...
│   ├── stemmed/
│   │   ├── model_labels.tex
│   │   ├── hdbscan_noise_coverage.tex
│   │   ├── fed_table_avg.tex
│   │   └── ...
│   └── no_stopword_removal/
│       ├── model_labels.tex
│       ├── hdbscan_noise_coverage.tex
│       ├── fed_table_avg.tex
│       └── ...
└── figures/
    ├── standard/
    ├── stemmed/
    └── no_stopword_removal/
```

---

## 3. HDBSCAN Noise Coverage LaTeX Table Fix

During LaTeX output generation for HDBSCAN noise coverage (`calculate_noise_coverage.py`), standard deviation expressions containing `\pm` were previously rendered without math mode delimiters (`$ ... $`). In LaTeX, `\pm` outside math mode causes a compilation error (`! Missing $ inserted.`).

This was resolved by enclosing the numeric mean and standard deviation within inline math mode:
```python
# Fixed string formatting in calculate_noise_coverage.py
pct_str = f"${mean_pct:.2f} \\pm {std_pct:.2f}$\\%"
```

---

## 4. Execution Commands

### Generate All Results (Standard, Stemmed, No-Stopwords)
To generate complete tables for all result types in the local output directory:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/pipelines/local_windows/get_results.ps1
```

### Export Directly to Dissertation Release Folder
To publish generated tables directly to the dissertation release folder (`-Release` switch):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/pipelines/local_windows/get_results.ps1 -Release
```

### Process a Single Result Type
To generate tables only for stemmed results:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/pipelines/local_windows/get_results.ps1 -ResultType stemmed
```
