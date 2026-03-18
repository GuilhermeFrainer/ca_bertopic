import pytest
from pathlib import Path
from src.utils import load_config

@pytest.fixture
def experiments_dir(tmp_path):
    """Creates a temporary experiments directory with base and inherited configs."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    
    # Create base dataset config
    base_config = {
        "dataset_path": "data/base.parquet",
        "embedding_col": "base_emb",
        "text_col": "base_text",
        "covariates": {"numerical": ["feat1"]}
    }
    with open(datasets_dir / "base.yaml", "w") as f:
        import yaml
        yaml.dump(base_config, f)
        
    # Create inheriting experiment config
    exp_config = {
        "extends": "datasets/base.yaml",
        "experiment": {
            "name": "inherited_exp",
            "sample_size": 500
        }
    }
    with open(tmp_path / "exp.yaml", "w") as f:
        yaml.dump(exp_config, f)
        
    return tmp_path

def test_load_config_inheritance(experiments_dir):
    """Tests that load_config correctly merges inherited values."""
    config = load_config("exp", experiments_dir)
    
    exp = config.get("experiment")
    assert exp is not None
    
    # Inherited values
    assert exp["dataset_path"] == "data/base.parquet"
    assert exp["embedding_col"] == "base_emb"
    
    # Overridden/New values
    assert exp["name"] == "inherited_exp"
    assert exp["sample_size"] == 500

def test_load_config_override(experiments_dir):
    """Tests that values in the experiment file override the base file."""
    # Create an experiment that overrides a base value
    override_config = {
        "extends": "datasets/base.yaml",
        "experiment": {
            "dataset_path": "data/override.parquet"
        }
    }
    with open(experiments_dir / "override.yaml", "w") as f:
        import yaml
        yaml.dump(override_config, f)
        
    config = load_config("override", experiments_dir)
    assert config["experiment"]["dataset_path"] == "data/override.parquet"

def test_load_config_missing_base(experiments_dir):
    """Tests that a missing base file raises FileNotFoundError."""
    broken_config = {"extends": "datasets/nonexistent.yaml"}
    with open(experiments_dir / "broken.yaml", "w") as f:
        import yaml
        yaml.dump(broken_config, f)
        
    with pytest.raises(FileNotFoundError):
        load_config("broken", experiments_dir)
