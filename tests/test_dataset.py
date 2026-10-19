import os
import pytest
from PIL import Image
import numpy as np
import tensorflow as tf

from src.dataset import get_datasets, get_val_dataset

@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    """Creates a miniature synthetic dataset with 2 classes and 4 images."""
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    for split_dir in [train_dir, val_dir]:
        for class_name in ["Aston Martin", "Audi Sedan"]:
            c_dir = split_dir / class_name
            c_dir.mkdir(parents=True)
            for i in range(2):
                img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
                img.save(c_dir / f"img_{i}.jpg")

    return {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "classes": ["Aston Martin", "Audi Sedan"]
    }

def test_get_datasets_shapes_and_classes(synthetic_dataset_dir):
    train_ds, val_ds, class_names = get_datasets(
        train_dir=synthetic_dataset_dir["train_dir"],
        val_dir=synthetic_dataset_dir["val_dir"],
        batch_size=2,
        image_size=(224, 224),
        seed=42
    )
    assert train_ds is not None
    assert val_ds is not None
    assert class_names == synthetic_dataset_dir["classes"]

    for images, labels in train_ds.take(1):
        assert images.shape == (2, 224, 224, 3)
        assert labels.shape == (2, 2)
        assert images.dtype == tf.float32

def test_val_dataset_unshuffled_ordering(synthetic_dataset_dir):
    """Confirms validation dataset returns deterministic batch ordering."""
    val_ds1, classes1 = get_val_dataset(synthetic_dataset_dir["val_dir"], batch_size=2)
    val_ds2, classes2 = get_val_dataset(synthetic_dataset_dir["val_dir"], batch_size=2)

    assert classes1 == classes2
    batch1_labels = [labels.numpy() for _, labels in val_ds1]
    batch2_labels = [labels.numpy() for _, labels in val_ds2]

    for l1, l2 in zip(batch1_labels, batch2_labels):
        np.testing.assert_array_equal(l1, l2)

def test_missing_dataset_directories():
    train_ds, val_ds, classes = get_datasets(
        train_dir="/non/existent/train",
        val_dir="/non/existent/val"
    )
    assert train_ds is None
    assert val_ds is None
    assert classes is None
