# CA-BERTopic Project

This project aims to modify BERTopic to incorporate document-level metadata into the topic modeling process.

## Getting Started

### Prerequisites

*   Python 3.12+
*   `uv` for package management.

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd CA-BERTopic
    ```

2.  Create a virtual environment and install the dependencies using `uv`:
    ```bash
    uv venv
    uv sync
    ```
    **Note:** This project may contain dependencies that are specific to a Linux environment with NVIDIA GPU support (e.g., `cudf`, `cuml`). On other operating systems like Windows or macOS, you may need to adjust the dependencies in `pyproject.toml`.

## Running Experiments

Experiments are defined by `.yaml` files in the `experiments/` directory and can be run using the `scripts/run_experiment.py` script. These experiments typically involve comparing multiple, fully-specified models.

To run an experiment, use the following command:
```bash
python scripts/run_experiment.py --exp <experiment_name>
```
Replace `<experiment_name>` with the name of the yaml file in the `experiments` directory (without the `.yaml` extension).

For example, to run the `trump.yaml` experiment:
```bash
python scripts/run_experiment.py --exp trump
```

### Running Hyperparameter Optimization

For hyperparameter tuning across a single model architecture, use the `scripts/run_optimizer.py` script. The setup is similar to a regular experiment, but the YAML file should contain a single `model` configuration. Within that configuration, any parameter that you want to search over should be specified as a list of values (e.g., `n_clusters: [50, 100, 150]`).

To run an optimization, use the following command:
```bash
python scripts/run_optimizer.py --exp <optimization_name>
```
For example, to run the `yelp_opt_spectral.yaml` optimization:
```bash
python scripts/run_optimizer.py --exp yelp_opt_spectral
```

## Visualizing Results

The project includes an interactive dashboard built with Streamlit to visualize and compare experiment results.

To run the dashboard:
```bash
uv run streamlit run scripts/dashboard.py
```

The dashboard allows you to:
*   Filter experiments by dataset, model type, date, and experiment type.
*   Compare models across various metrics (e.g., `u_mass`, `irbo`, `c_v`).
*   Automatically highlight best-performing models in the results table.
*   Identify trends using dynamic scatter plots.

## Running Tests

To run the test suite, use the following command:
```bash
uv run -m pytest
```

## Project Structure

```
├── data/              # Raw and processed datasets
├── experiments/       # Experiment configuration files
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── results/           # Experiment results
├── scripts/           # Utility scripts
│   ├── build_yelp_dataset.py
│   ├── dashboard.py       # Interactive results dashboard
│   ├── generate_embeddings.py
│   ├── preprocess_datasets.py
│   ├── run_experiment.py
│   └── run_optimizer.py
├── src/               # Source code
│   ├── data.py        # Data loading and preprocessing
│   ├── models.py      # Custom model definitions
│   ├── optimizer.py   # Hyperparameter optimization
│   ├── training.py    # Training scripts
│   ├── evaluation.py  # Evaluation metrics
│   ├── processing.py  # Data processing utilities
│   ├── embeddings.py  # Embedding generation
│   ├── mvc_wrapper.py # Multi-view clustering wrapper
│   ├── make_table.py  # Result table generation
│   ├── logger_config.py
│   └── utils.py       # General utilities
└── pyproject.toml     # Project configuration and dependencies
```
