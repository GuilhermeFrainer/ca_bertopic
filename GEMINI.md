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
│   ├── training.py    # Training scripts
│   └── evaluation.py  # Evaluation metrics
├── run_experiment.py  # Main script to run experiments
└── pyproject.toml     # Project configuration and dependencies
```

## Core Technologies

*   **BERTopic:** For topic modeling.
*   **PyTorch:** As a backend for deep learning models.
*   **Polars:** For data manipulation.
*   **scikit-learn:** For machine learning utilities.
*   **Great Tables:** For creating publication-ready tables of results.
*   **uv:** For python package management.
