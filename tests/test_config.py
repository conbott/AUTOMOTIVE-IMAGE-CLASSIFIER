import pytest
import tempfile
import yaml
import os

from src.config import load_config, DEFAULT_CONFIG_PATH

def test_load_default_config():
    config = load_config()
    assert isinstance(config, dict)
    assert "seed" in config
    assert isinstance(config["seed"], int)
    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert "output" in config
    assert "mlflow" in config
    assert config["data"]["batch_size"] > 0
    assert config["model"]["num_classes"] == 196
    assert config["data"]["image_size"] == [224, 224]

def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/non/existent/path/config.yaml")

def test_missing_required_section(tmp_path):
    bad_config = {
        "seed": 42,
        "data": {"train_dir": "train", "val_dir": "val", "batch_size": 32, "image_size": [224, 224]},
        # Missing 'model', 'training', 'output'
    }
    cfg_file = tmp_path / "bad_config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(bad_config, f)

    with pytest.raises(ValueError, match="Missing required configuration section"):
        load_config(str(cfg_file))

def test_invalid_seed_type(tmp_path):
    bad_config = {
        "seed": "not_an_int",
        "data": {"train_dir": "train", "val_dir": "val", "batch_size": 32, "image_size": [224, 224]},
        "model": {"num_classes": 196},
        "training": {},
        "output": {}
    }
    cfg_file = tmp_path / "bad_seed.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(bad_config, f)

    with pytest.raises(AssertionError, match="seed must be an integer"):
        load_config(str(cfg_file))

def test_mlflow_defaults_when_omitted(tmp_path):
    cfg_without_mlflow = {
        "seed": 42,
        "data": {"train_dir": "train", "val_dir": "val", "batch_size": 32, "image_size": [224, 224]},
        "model": {"num_classes": 196},
        "training": {},
        "output": {}
    }
    cfg_file = tmp_path / "no_mlflow.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(cfg_without_mlflow, f)

    loaded = load_config(str(cfg_file))
    assert "mlflow" in loaded
    assert loaded["mlflow"]["tracking_uri"] == "mlruns"
    assert loaded["mlflow"]["experiment_name"] == "automotive-image-classifier"
