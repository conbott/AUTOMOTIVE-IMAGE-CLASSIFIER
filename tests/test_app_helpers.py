import io
import json
import pytest
from PIL import Image
import numpy as np
import tensorflow as tf

from app.utils import (
    preprocess_image,
    format_predictions,
    load_model_and_classes,
    validate_image,
    get_model_metadata,
    get_file_mtime
)

def test_preprocess_image():
    # Test with arbitrary size RGBA image
    img = Image.new("RGBA", (150, 100), color=(255, 0, 0, 255))
    batch = preprocess_image(img, target_size=(224, 224))

    assert batch.shape == (1, 224, 224, 3)
    assert isinstance(batch, np.ndarray)
    # Range should be [0, 255] float
    assert batch.min() >= 0.0
    assert batch.max() <= 255.0

def test_format_predictions_sorting_and_confidence():
    classes = ["Audi A4", "BMW 3 Series", "Mercedes C-Class", "Tesla Model 3"]
    # Model softmax probabilities
    probs = np.array([0.05, 0.70, 0.15, 0.10])

    top_3 = format_predictions(probs, classes, top_k=3)
    assert len(top_3) == 3
    assert top_3[0] == ("BMW 3 Series", 70.0)
    assert top_3[1] == ("Mercedes C-Class", 15.0)
    assert top_3[2] == ("Tesla Model 3", 10.0)

def test_format_predictions_dimension_mismatch():
    classes = ["Audi", "BMW"]
    probs = np.array([0.5, 0.3, 0.2]) # 3 probs for 2 classes
    with pytest.raises(ValueError, match="Dimension mismatch"):
        format_predictions(probs, classes)

def test_load_model_and_classes_graceful_missing(tmp_path):
    model, classes = load_model_and_classes(
        model_path=str(tmp_path / "nonexistent.keras"),
        classes_path=str(tmp_path / "nonexistent.txt")
    )
    assert model is None
    assert classes is None

def test_load_model_and_classes_valid(tmp_path):
    # Save miniature valid model and classes
    inputs = tf.keras.Input(shape=(224, 224, 3))
    outputs = tf.keras.layers.Dense(2, activation="softmax")(inputs)
    m = tf.keras.Model(inputs=inputs, outputs=outputs)
    m_path = tmp_path / "m.keras"
    m.save(str(m_path))

    c_path = tmp_path / "c.txt"
    c_path.write_text("Car1\nCar2\n")

    loaded_m, loaded_c = load_model_and_classes(str(m_path), str(c_path))
    assert loaded_m is not None
    assert loaded_c == ["Car1", "Car2"]

def test_validate_image_valid():
    # In-memory JPEG
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    is_valid, out_img, err = validate_image(buf)
    assert is_valid is True
    assert out_img is not None
    assert err is None
    assert out_img.size == (64, 64)

def test_validate_image_corrupt():
    corrupt_buf = io.BytesIO(b"this is totally corrupted image binary data")
    is_valid, out_img, err = validate_image(corrupt_buf)
    assert is_valid is False
    assert out_img is None
    assert "Unable to read the uploaded image" in err

def test_validate_image_none():
    is_valid, out_img, err = validate_image(None)
    assert is_valid is False
    assert out_img is None
    assert "No image file provided" in err

def test_get_file_mtime(tmp_path):
    sample = tmp_path / "file.txt"
    sample.write_text("hello")
    mtime = get_file_mtime(str(sample))
    assert mtime > 0.0

    nonexistent_mtime = get_file_mtime(str(tmp_path / "nonexistent.txt"))
    assert nonexistent_mtime == 0.0

def test_get_model_metadata_baseline(tmp_path):
    baseline_file = tmp_path / "baseline_manifest.json"
    manifest_data = {
        "macro_f1": 0.5698,
        "accuracy": 0.5734,
        "loss": 1.6796,
        "total_samples": 1629,
        "num_classes": 196
    }
    baseline_file.write_text(json.dumps(manifest_data))

    meta = get_model_metadata(models_dir=str(tmp_path))
    assert meta["is_available"] is True
    assert meta["status_label"] == "Verified Baseline"
    assert meta["metrics"]["macro_f1"] == 0.5698
    assert "1,629" in meta["evaluation_context"]

def test_get_model_metadata_promoted(tmp_path):
    serving_file = tmp_path / "serving_manifest.json"
    manifest_data = {
        "run_id": "run_candidate_2026",
        "metrics": {"macro_f1": 0.6500, "accuracy": 0.6800, "loss": 1.3500}
    }
    serving_file.write_text(json.dumps(manifest_data))

    meta = get_model_metadata(models_dir=str(tmp_path))
    assert meta["is_available"] is True
    assert meta["status_label"] == "Promoted Model"
    assert meta["version_or_run_id"] == "run_candidate_2026"
    assert meta["metrics"]["macro_f1"] == 0.65

def test_get_model_metadata_fallback(tmp_path):
    meta = get_model_metadata(models_dir=str(tmp_path))
    assert meta["is_available"] is False
    assert meta["version_or_run_id"] == "Version metadata unavailable."
    assert meta["metrics"] is None
