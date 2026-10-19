import os
import yaml
from typing import Dict, Any

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "train_config.yaml"
)

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads and validates the YAML training configuration.
    Falls back to DEFAULT_CONFIG_PATH if no path is provided.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate essential keys
    required_sections = ["seed", "data", "model", "training", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: '{section}'")

    # Validate specific fields
    assert isinstance(config["seed"], int), "seed must be an integer"
    assert "train_dir" in config["data"], "data.train_dir is required"
    assert "val_dir" in config["data"], "data.val_dir is required"
    assert len(config["data"]["image_size"]) == 2, "image_size must have 2 dimensions (H, W)"
    assert config["data"]["batch_size"] > 0, "batch_size must be positive"
    assert config["model"]["num_classes"] > 0, "num_classes must be positive"

    # Default MLflow section if missing
    if "mlflow" not in config:
        config["mlflow"] = {
            "tracking_uri": "mlruns",
            "experiment_name": "automotive-image-classifier"
        }

    return config

if __name__ == "__main__":
    cfg = load_config()
    print("Configuration loaded successfully:")
    print(yaml.dump(cfg, default_flow_style=False))
