import os
import json
import pytest
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.evaluate import evaluate_model

def test_metrics_calculation_known_vectors():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2]) # 1 error at index 2 (predicted 1 instead of 2)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_prec = precision_score(y_true, y_pred, average="macro")
    macro_rec = recall_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    assert round(acc, 4) == round(5 / 6, 4)
    assert 0.0 < macro_f1 < 1.0
    assert cm.shape == (3, 3)
    assert cm.sum() == 6

def test_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    assert accuracy_score(y_true, y_pred) == 1.0
    assert f1_score(y_true, y_pred, average="macro") == 1.0
    assert precision_score(y_true, y_pred, average="macro") == 1.0
    assert recall_score(y_true, y_pred, average="macro") == 1.0

def test_evaluate_model_synthetic_pipeline(tmp_path):
    # Setup miniature 2-class dataset
    val_dir = tmp_path / "val"
    for c_name in ["ClassA", "ClassB"]:
        c_dir = val_dir / c_name
        c_dir.mkdir(parents=True)
        for i in range(2):
            img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
            img.save(c_dir / f"sample_{i}.jpg")

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("ClassA\nClassB\n")

    # Create small model with 2 outputs
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model_path = tmp_path / "mini_model.keras"
    model.save(str(model_path))

    out_dir = tmp_path / "eval_out"
    metrics = evaluate_model(
        model_path=str(model_path),
        val_dir=str(val_dir),
        classes_path=str(classes_file),
        output_dir=str(out_dir),
        batch_size=2,
        manifest_name="test_metrics.json"
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["total_samples"] == 4
    assert metrics["num_classes"] == 2
    assert os.path.exists(out_dir / "test_metrics.json")
    assert os.path.exists(out_dir / "confusion_matrix.json")
