import tensorflow as tf
import os
from typing import Tuple, List, Optional

def get_datasets(
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 42
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[List[str]]]:
    """
    Loads the processed Stanford Cars dataset and returns tf.data.Dataset objects for train and validation sets.
    Validation dataset is intentionally un-shuffled (shuffle=False) for deterministic evaluation.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if train_dir is None:
        if data_dir is not None:
            train_dir = os.path.join(data_dir, 'train')
        else:
            train_dir = os.path.join(base_dir, 'data', 'processed', 'train')
            
    if val_dir is None:
        if data_dir is not None:
            val_dir = os.path.join(data_dir, 'val')
        else:
            val_dir = os.path.join(base_dir, 'data', 'processed', 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        print("Please run `python src/prepare_data.py` first to organize the dataset.")
        return None, None, None

    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return None, None, None

    # Load training dataset (shuffled with fixed seed)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=seed,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Load validation dataset (un-shuffled for deterministic evaluation)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        shuffle=False,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_names = train_ds.class_names

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    # Data augmentation for training only
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.1, seed=seed),
        tf.keras.layers.RandomZoom(0.1, seed=seed),
    ], name="data_augmentation")

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )
    
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def get_val_dataset(
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[Optional[tf.data.Dataset], Optional[List[str]]]:
    """
    Loads only the validation dataset without training data overhead.
    Un-shuffled for reproducible evaluation metrics.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if val_dir is None:
        val_dir = os.path.join(base_dir, 'data', 'processed', 'val')

    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return None, None

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        shuffle=False,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    class_names = val_ds.class_names
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return val_ds, class_names

if __name__ == "__main__":
    print("Testing dataset loader...")
    train_ds, val_ds, classes = get_datasets(batch_size=16)
    if train_ds is not None:
        print(f"Number of classes: {len(classes)}")
        for img_b, lbl_b in train_ds.take(1):
            print(f"Train batch shape: images {img_b.shape}, labels {lbl_b.shape}")
        for img_b, lbl_b in val_ds.take(1):
            print(f"Val batch shape: images {img_b.shape}, labels {lbl_b.shape}")
