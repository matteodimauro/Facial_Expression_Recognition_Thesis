from tensorflow import keras
import tensorflow as tf
from dataset_loader import FRAMES_TOTAL, NUM_LANDMARKS, CLASSES


# Utility for our sequence model.
def get_conv_lstm_model():
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer(input_shape=(FRAMES_TOTAL, NUM_LANDMARKS * 3),
                                    name="frame_features_input"),
         tf.keras.layers.Conv1D(kernel_size=5, filters=10, use_bias=False, padding='causal'),
         tf.keras.layers.BatchNormalization(center=True, scale=False),
         tf.keras.layers.Activation('relu'),
         tf.keras.layers.LSTM(10, use_bias=False, return_sequences=True),
         tf.keras.layers.BatchNormalization(center=True, scale=False),
         tf.keras.layers.Activation('relu'),

         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(8, use_bias=False),
         tf.keras.layers.BatchNormalization(center=True, scale=False),
         tf.keras.layers.Activation('relu'),
         tf.keras.layers.Dropout(0.3),
         tf.keras.layers.Dense(len(CLASSES), activation='softmax')
         ])
    optimizer = keras.optimizers.SGD(learning_rate=0.00001)
    loss_fn = keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=accuracy)

    return model
