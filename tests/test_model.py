import pytest
import numpy as np
import tensorflow as tf

from src.model import build_model, unfreeze_model

def test_build_model_architecture():
    num_classes = 196
    model, base_model = build_model(num_classes=num_classes, image_size=(224, 224), learning_rate=0.001)

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 196)

    # Base model should be frozen
    assert base_model.trainable is False

    # Check output activation
    last_layer = model.layers[-1]
    assert isinstance(last_layer, tf.keras.layers.Dense)
    assert last_layer.activation == tf.keras.activations.softmax
    assert last_layer.units == num_classes

    # Optimizer and loss
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert model.loss == "categorical_crossentropy"

def test_model_forward_pass_synthetic():
    num_classes = 10
    model, _ = build_model(num_classes=num_classes, image_size=(224, 224))
    synthetic_batch = np.random.uniform(0, 255, (2, 224, 224, 3)).astype(np.float32)
    preds = model.predict(synthetic_batch, verbose=0)

    assert preds.shape == (2, num_classes)
    # Output should sum to 1.0 per sample (valid softmax probabilities)
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(2), rtol=1e-5)

def test_unfreeze_model():
    model, base_model = build_model(num_classes=196)
    unfrozen_model = unfreeze_model(model, base_model, learning_rate=1e-5)

    assert base_model.trainable is True
    # BatchNormalization layers must remain frozen
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False
