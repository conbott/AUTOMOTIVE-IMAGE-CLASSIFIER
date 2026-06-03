import os
import tensorflow as tf
from dataset import get_datasets
from model import build_model, unfreeze_model

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    model_save_path = os.path.join(base_dir, 'models', 'car_classifier.keras')
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, class_names = get_datasets(data_dir=data_dir, batch_size=32)
    
    if train_ds is None:
        return
        
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")

    print("Building model...")
    model, base_model = build_model(num_classes=num_classes)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, monitor='val_accuracy', save_best_only=True
    )

    print("Starting initial training (top layers only)...")
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )

    print("Unfreezing base model for fine-tuning...")
    model = unfreeze_model(model, base_model)

    print("Starting fine-tuning...")
    fine_tune_epochs = 10
    total_epochs = epochs + fine_tune_epochs

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        callbacks=[early_stopping, model_checkpoint]
    )

    print(f"Training complete. Best model saved to {model_save_path}")

    # Optionally, save the class names
    with open(os.path.join(os.path.dirname(model_save_path), 'classes.txt'), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

if __name__ == "__main__":
    train()
