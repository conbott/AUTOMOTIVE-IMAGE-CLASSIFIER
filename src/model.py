import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_model(num_classes, image_size=(224, 224), learning_rate=0.001):
    """
    Builds a transfer learning model based on EfficientNetB0.
    """
    # Load pre-trained EfficientNetB0 without the top classification layer
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(image_size[0], image_size[1], 3)
    )

    # Freeze the base model to prevent destroying pre-trained weights during early training
    base_model.trainable = False

    # Build the complete model
    model = models.Sequential([
        # Optionally, you can add data augmentation here as a layer, 
        # but we already added it to the dataset pipeline in dataset.py
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def unfreeze_model(model, base_model, learning_rate=1e-5):
    """
    Unfreezes the base model for fine-tuning with a lower learning rate.
    """
    # Unfreeze the base model
    base_model.trainable = True

    # We typically keep BatchNormalization layers frozen during fine-tuning
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Recompile with a very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("Testing model build...")
    # 196 classes for Stanford Cars
    model, base_model = build_model(num_classes=196)
    model.summary()
