from tensorflow import keras
import tensorflow as tf
from dataset_loader import FRAMES_TOTAL, NUM_LANDMARKS, CLASSES


# Utility for our sequence model.
def get_conv_model():
    model = tf.keras.Sequential(

        [tf.keras.layers.InputLayer(input_shape=(FRAMES_TOTAL, NUM_LANDMARKS * 3),
                                    name="frame_features_input"),

         tf.keras.layers.Reshape(input_shape=(FRAMES_TOTAL, NUM_LANDMARKS * 3),
                                 target_shape=(FRAMES_TOTAL, NUM_LANDMARKS * 3, 1)),

         tf.keras.layers.Conv2D(kernel_size=3, filters=12, use_bias=False, padding='same'),
         tf.keras.layers.BatchNormalization(center=True, scale=False),
         tf.keras.layers.Activation('relu'),

         tf.keras.layers.Conv2D(kernel_size=5, filters=24, use_bias=False, padding='same', strides=2),
         tf.keras.layers.BatchNormalization(center=True, scale=False),
         tf.keras.layers.Activation('relu'),

         tf.keras.layers.Conv2D(kernel_size=5, filters=32, use_bias=False, padding='same', strides=2),
         tf.keras.layers.BatchNormalization(center=True, scale=False),
         tf.keras.layers.Activation('relu'),

         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(200, use_bias=False),
         tf.keras.layers.BatchNormalization(center=True, scale=False),
         tf.keras.layers.Activation('relu'),

         tf.keras.layers.Dropout(0.3),
         tf.keras.layers.Dense(len(CLASSES), activation='softmax')

         ])
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=accuracy)

    return model
