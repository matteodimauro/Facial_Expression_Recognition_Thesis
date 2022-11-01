from tensorflow import keras
import tensorflow as tf
import os
import cremad
import oulu_casia
import ckplus
from dataset_loader import FRAMES_TOTAL, NUM_LANDMARKS, CLASSES


# Deep Temporal Geometry Network
def get_dtgm_model(dataset):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(dataset.FRAMES_TOTAL, dataset.NUM_LANDMARKS * 3),
                                   name="frame_features_input"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, use_bias=False, activation='relu'),
        tf.keras.layers.Dense(600, use_bias=False, activation='relu'),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(dataset.CLASSES), activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
    loss_fn = keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=accuracy)

    return model
