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
    **Note:** This project contains dependencies that are specific to a Linux environment with NVIDIA GPU support (e.g., `cudf`, `cuml`). On other operating systems like Windows or macOS, you may need to adjust the dependencies in `pyproject.toml`.

## Running Experiments

Experiments are defined by `.yaml` files in the `experiments/` directory and can be run using the `run_experiment.py` script.

To run an experiment, use the following command:
```bash
python run_experiment.py --exp <experiment_name>
```
Replace `<experiment_name>` with the name of the yaml file in the `experiments` directory (without the `.yaml` extension).

For example, to run the `trump.yaml` experiment:
```bash
python run_experiment.py --exp trump
```

### Running Hyperparameter Optimization

For hyperparameter tuning, use the `run_optimizer.py` script. The setup is similar to a regular experiment, but the YAML file should contain a single `model` configuration. Within that configuration, any parameter that you want to search over should be specified as a list of values.

To run an optimization, use the following command:
```bash
python run_optimizer.py --exp <optimization_name>
```
For example, to run the `yelp_opt_spectral.yaml` optimization:
```bash
python run_optimizer.py --exp yelp_opt_spectral
```

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
├── src/               # Source code
│   ├── data.py        # Data loading and preprocessing
│   ├── models.py      # Custom model definitions
│   ├── optimizer.py   # Hyperparameter optimization
│   ├── training.py    # Training scripts
│   └── evaluation.py  # Evaluation metrics
├── run_experiment.py  # Main script to run experiments
├── run_optimizer.py   # Main script to run hyperparameter optimization
└── pyproject.toml     # Project configuration and dependencies
```

## Core Technologies

*   **BERTopic:** For topic modeling.
*   **PyTorch:** As a backend for deep learning models.
*   **Polars:** For data manipulation.
*   **scikit-learn:** For machine learning utilities.
*   **Great Tables:** For creating publication-ready tables of results.
*   **uv:** For python package management.

## Coding Style

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
All Python code should adhere to these conventions.
