import tensorflow as tf
import keras

NUM_LABELS = 99
IMAGE_SIZE = (224, 224)

# Data augmentation
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomBrightness((-0.3, 0.3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom((-0.2, 0.2)),
        tf.keras.layers.RandomRotation(0.1),
    ],
    name="augmentation",
)


# Create the CNN model
def create_model():
    model = tf.keras.Sequential(
        name="CNN_model",
        layers=[
            # Input
            keras.Input(shape=IMAGE_SIZE + (1,), name="input"),
            # Data augmentation
            data_augmentation,
            # Normalization
            keras.layers.Rescaling(1.0 / 255),
            # Convolution layers
            # Block 1
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            # Block 2
            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            # Block 3
            keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Dropout(0.05),
            # Block 4
            keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Dropout(0.1),
            # Flatten and Fully Connected
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.65),
            keras.layers.Dense(NUM_LABELS, activation="softmax", name="output"),
        ],
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


weights_path = "./Models/cnn_weights_2.keras"
model_path = "./Models/cnn_model_2.keras"
model = create_model()
model.load_weights(weights_path)
model.summary()
model.save(model_path)
