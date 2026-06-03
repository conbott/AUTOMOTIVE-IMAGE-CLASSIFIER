import tensorflow as tf
import os

def get_datasets(data_dir='./data/processed', batch_size=32, image_size=(224, 224)):
    """
    Loads the processed Stanford Cars dataset and returns tf.data.Dataset objects for train and validation sets.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        print("Please run `python src/prepare_data.py` first to organize the dataset.")
        return None, None, None

    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_names = train_ds.class_names

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    print("Testing dataset loader...")
    # Update path since script is in src/ but runs from root ideally
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    train_ds, val_ds, classes = get_datasets(data_dir=processed_dir)
    
    if train_ds:
        print(f"Number of classes: {len(classes)}")
        print(f"First 5 classes: {classes[:5]}")
        
        for image_batch, labels_batch in train_ds.take(1):
            print(f"Batch images shape: {image_batch.shape}")
            print(f"Batch labels shape: {labels_batch.shape}")
