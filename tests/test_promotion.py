import json
import os
import shutil
import tempfile
import pytest
import numpy as np
import tensorflow as tf

from src.promote import (
    calculate_sha256,
    validate_model_integrity,
    record_promotion_event,
    promote_candidate,
    rollback_model,
    get_baseline_macro_f1
)

@pytest.fixture
def mock_env(tmp_path):
    """
    Creates an isolated mock environment with a baseline manifest,
    a mock serving model, and temporary artifact directories.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    backups_dir = models_dir / "backups"
    backups_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts" / "runs"
    artifacts_dir.mkdir(parents=True)

    # Create a small valid Keras model (196 classes, 224x224x3 input)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(196, activation="softmax")(x)
    dummy_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    serving_model_path = models_dir / "car_classifier.keras"
    dummy_model.save(str(serving_model_path))

    baseline_backup_path = models_dir / "car_classifier_baseline_backup.keras"
    dummy_model.save(str(baseline_backup_path))

    # Baseline manifest
    baseline_manifest = {
        "model_name": "car_classifier_baseline",
        "macro_f1": 0.5698,
        "accuracy": 0.5734,
        "loss": 1.6796,
        "num_classes": 196
    }
    with open(models_dir / "baseline_manifest.json", "w") as f:
        json.dump(baseline_manifest, f)

    return {
        "tmp_path": tmp_path,
        "models_dir": models_dir,
        "backups_dir": backups_dir,
        "artifacts_dir": artifacts_dir,
        "serving_model_path": str(serving_model_path),
        "baseline_backup_path": str(baseline_backup_path),
        "dummy_model": dummy_model
    }

def test_calculate_sha256(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Automotive Classifier DLOps")
    h1 = calculate_sha256(str(test_file))
    h2 = calculate_sha256(str(test_file))
    assert h1 == h2
    assert len(h1) == 64

def test_validate_model_integrity_valid(mock_env):
    is_valid, err = validate_model_integrity(mock_env["serving_model_path"])
    assert is_valid is True
    assert err is None

def test_validate_model_integrity_invalid_shape(tmp_path):
    # Model with wrong output shape (10 classes instead of 196)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    wrong_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    wrong_path = tmp_path / "wrong_model.keras"
    wrong_model.save(str(wrong_path))

    is_valid, err = validate_model_integrity(str(wrong_path), expected_output_classes=196)
    assert is_valid is False
    assert "Output classes mismatch" in err

def test_quality_gate_rejects_inferior_candidate(mock_env, monkeypatch):
    run_id = "run_inferior_candidate"
    run_dir = mock_env["artifacts_dir"] / run_id
    run_dir.mkdir()

    # Save candidate model
    mock_env["dummy_model"].save(str(run_dir / "candidate_model.keras"))
    # Inferior F1: 0.5200 < 0.5698
    metrics = {"macro_f1": 0.5200, "accuracy": 0.5300, "loss": 1.8000}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump({"run_id": run_id}, f)
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump({"total_classes": 196}, f)

    # Monkeypatch paths
    monkeypatch.setattr("src.promote.PROJECT_ROOT", str(mock_env["tmp_path"]))
    monkeypatch.setattr("src.promote.BASELINE_MANIFEST_PATH", str(mock_env["models_dir"] / "baseline_manifest.json"))
    monkeypatch.setattr("src.promote.SERVING_MODEL_PATH", mock_env["serving_model_path"])
    monkeypatch.setattr("src.promote.HISTORY_FILE_PATH", str(mock_env["models_dir"] / "promotion_history.json"))

    original_hash = calculate_sha256(mock_env["serving_model_path"])
    promoted = promote_candidate(run_id, serving_model_path=mock_env["serving_model_path"])

    assert promoted is False
    # Verify serving model was unchanged
    assert calculate_sha256(mock_env["serving_model_path"]) == original_hash

def test_quality_gate_accepts_superior_candidate(mock_env, monkeypatch):
    run_id = "run_superior_candidate"
    run_dir = mock_env["artifacts_dir"] / run_id
    run_dir.mkdir()

    # Create a distinct model for candidate
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(196, activation="softmax")(x)
    superior_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    superior_model.save(str(run_dir / "candidate_model.keras"))

    # Superior F1: 0.6500 > 0.5698
    metrics = {"macro_f1": 0.6500, "accuracy": 0.6600, "loss": 1.4000}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump({"run_id": run_id}, f)
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump({"total_classes": 196}, f)

    monkeypatch.setattr("src.promote.PROJECT_ROOT", str(mock_env["tmp_path"]))
    monkeypatch.setattr("src.promote.BASELINE_MANIFEST_PATH", str(mock_env["models_dir"] / "baseline_manifest.json"))
    monkeypatch.setattr("src.promote.SERVING_MODEL_PATH", mock_env["serving_model_path"])
    monkeypatch.setattr("src.promote.SERVING_MANIFEST_PATH", str(mock_env["models_dir"] / "serving_manifest.json"))
    monkeypatch.setattr("src.promote.HISTORY_FILE_PATH", str(mock_env["models_dir"] / "promotion_history.json"))
    monkeypatch.setattr("src.promote.BACKUPS_DIR", str(mock_env["backups_dir"]))

    candidate_hash = calculate_sha256(str(run_dir / "candidate_model.keras"))
    pre_serving_hash = calculate_sha256(mock_env["serving_model_path"])

    promoted = promote_candidate(run_id, serving_model_path=mock_env["serving_model_path"])

    assert promoted is True
    # Verify serving model now has the candidate hash
    assert calculate_sha256(mock_env["serving_model_path"]) == candidate_hash
    assert calculate_sha256(mock_env["serving_model_path"]) != pre_serving_hash

def test_missing_files_rejection(mock_env, monkeypatch):
    run_id = "run_missing_files"
    run_dir = mock_env["artifacts_dir"] / run_id
    run_dir.mkdir()
    # Missing candidate_model.keras and metrics.json

    monkeypatch.setattr("src.promote.PROJECT_ROOT", str(mock_env["tmp_path"]))
    monkeypatch.setattr("src.promote.BASELINE_MANIFEST_PATH", str(mock_env["models_dir"] / "baseline_manifest.json"))
    monkeypatch.setattr("src.promote.SERVING_MODEL_PATH", mock_env["serving_model_path"])
    monkeypatch.setattr("src.promote.HISTORY_FILE_PATH", str(mock_env["models_dir"] / "promotion_history.json"))

    original_hash = calculate_sha256(mock_env["serving_model_path"])
    promoted = promote_candidate(run_id, serving_model_path=mock_env["serving_model_path"])
    assert promoted is False
    assert calculate_sha256(mock_env["serving_model_path"]) == original_hash

def test_rollback_restores_serving_model(mock_env, monkeypatch):
    monkeypatch.setattr("src.promote.PROJECT_ROOT", str(mock_env["tmp_path"]))
    monkeypatch.setattr("src.promote.SERVING_MODEL_PATH", mock_env["serving_model_path"])
    monkeypatch.setattr("src.promote.BACKUPS_DIR", str(mock_env["backups_dir"]))
    monkeypatch.setattr("src.promote.HISTORY_FILE_PATH", str(mock_env["models_dir"] / "promotion_history.json"))
    monkeypatch.setattr("src.promote.ORIGINAL_BASELINE_BACKUP", mock_env["baseline_backup_path"])

    # First alter the serving model to simulate a broken update
    with open(mock_env["serving_model_path"], "wb") as f:
        f.write(b"corrupt_model_bytes")

    # Roll back to the verified baseline backup
    success = rollback_model(backup_path=mock_env["baseline_backup_path"], serving_model_path=mock_env["serving_model_path"])
    assert success is True

    # Validate model is restored and working
    is_valid, err = validate_model_integrity(mock_env["serving_model_path"])
    assert is_valid is True
    assert calculate_sha256(mock_env["serving_model_path"]) == calculate_sha256(mock_env["baseline_backup_path"])
    # Confirm baseline backup was never deleted
    assert os.path.exists(mock_env["baseline_backup_path"])

def test_quality_gate_accepts_equal_candidate(mock_env, monkeypatch):
    run_id = "run_equal_candidate"
    run_dir = mock_env["artifacts_dir"] / run_id
    run_dir.mkdir()

    # Exact baseline F1: 0.5698
    mock_env["dummy_model"].save(str(run_dir / "candidate_model.keras"))
    metrics = {"macro_f1": 0.5698, "accuracy": 0.5734, "loss": 1.6796}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump({"run_id": run_id}, f)
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump({"total_classes": 196}, f)

    monkeypatch.setattr("src.promote.PROJECT_ROOT", str(mock_env["tmp_path"]))
    monkeypatch.setattr("src.promote.BASELINE_MANIFEST_PATH", str(mock_env["models_dir"] / "baseline_manifest.json"))
    monkeypatch.setattr("src.promote.SERVING_MODEL_PATH", mock_env["serving_model_path"])
    monkeypatch.setattr("src.promote.SERVING_MANIFEST_PATH", str(mock_env["models_dir"] / "serving_manifest.json"))
    monkeypatch.setattr("src.promote.HISTORY_FILE_PATH", str(mock_env["models_dir"] / "promotion_history.json"))
    monkeypatch.setattr("src.promote.BACKUPS_DIR", str(mock_env["backups_dir"]))

    promoted = promote_candidate(run_id, serving_model_path=mock_env["serving_model_path"])
    assert promoted is True

def test_malformed_manifest_rejection(mock_env, monkeypatch):
    run_id = "run_malformed_manifest"
    run_dir = mock_env["artifacts_dir"] / run_id
    run_dir.mkdir()

    mock_env["dummy_model"].save(str(run_dir / "candidate_model.keras"))
    # Missing macro_f1
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"accuracy": 0.60}, f)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump({"run_id": run_id}, f)
    with open(run_dir / "class_mapping.json", "w") as f:
        json.dump({"total_classes": 196}, f)

    monkeypatch.setattr("src.promote.PROJECT_ROOT", str(mock_env["tmp_path"]))
    monkeypatch.setattr("src.promote.BASELINE_MANIFEST_PATH", str(mock_env["models_dir"] / "baseline_manifest.json"))
    monkeypatch.setattr("src.promote.SERVING_MODEL_PATH", mock_env["serving_model_path"])
    monkeypatch.setattr("src.promote.HISTORY_FILE_PATH", str(mock_env["models_dir"] / "promotion_history.json"))

    original_hash = calculate_sha256(mock_env["serving_model_path"])
    promoted = promote_candidate(run_id, serving_model_path=mock_env["serving_model_path"])
    assert promoted is False
    assert calculate_sha256(mock_env["serving_model_path"]) == original_hash
