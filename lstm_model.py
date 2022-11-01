from tensorflow import keras
import tensorflow as tf
from dataset_loader import FRAMES_TOTAL, NUM_LANDMARKS, CLASSES


def get_lstm_model():
    model = tf.keras.Sequential([
        # tf.keras.layers.Reshape(input_shape=(),
        # target_shape=(16, FRAMES_TOTAL, NUM_LANDMARKS * 3)),

        tf.keras.layers.InputLayer(input_shape=( FRAMES_TOTAL, NUM_LANDMARKS * 3),
                                   name="frame_features_input"),

        tf.keras.layers.LSTM(15, use_bias=False, name="first_lstm", return_sequences=True),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.LSTM(30, use_bias=False, name="second_lstm", return_sequences=True),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.LSTM(25, use_bias=False, name="third_lstm", return_sequences=True),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(516, use_bias=False),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=accuracy)

    return model