# src/models.py
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from src.config import IMG_SIZE

def build_mobilenet_model(num_classes: int = 3, trainable_base: bool = False):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # 1. On-the-fly Data Augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomBrightness(0.1)(x)

    # 2. Backbone Preprocessing (Scales pixels to [-1, 1])
    x = applications.mobilenet_v2.preprocess_input(x)

    # 3. Backbone Feature Extractor
    base_model = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = trainable_base

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # 4. Classification Output Layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="DriverState_MobileNetV2")
    return model