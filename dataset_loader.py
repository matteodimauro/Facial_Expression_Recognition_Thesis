import numpy as np
import os
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FRAMES_TOTAL = 85
NUM_LANDMARKS = 478
NOSE_TIP_IDX = 19
STD_NORM = True

CLASSES = {"ANG": 0,
           "DIS": 1,
           "FEA": 2,
           "HAP": 3,
           "NEU": 4,
           "SAD": 5
           }

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 30


# function made on purpose to achieve a basic normalization
def NormalizeData(data, frames):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    start = 0
    end = NUM_LANDMARKS
    for i in range(0, frames):
        data[start:end, 0] = (data[start:end, 0] - data[start + 19][0])  # / np.std(data[:, 0])
        data[start:end, 1] = (data[start:end, 1] - data[start + 19][1])  # / np.std(data[:, 1])
        data[start:end, 2] = (data[start:end, 2] - data[start + 19][2])  # / np.std(data[:, 2])
        if start + end < data.shape[0]:
            start = end + 1
            end = start + end
    return scaler.fit_transform(data)


def extract_label(path):
    parts = tf.strings.split(path, os.path.sep)
    one_hot = tf.strings.split(parts[-1], '_')[-2] == tf.constant(list(CLASSES.keys()))

    return tf.cast(one_hot, tf.int8)


def _fixup_shape(images, labels):
    images.set_shape([FRAMES_TOTAL, NUM_LANDMARKS * 3])
    labels.set_shape([len(CLASSES)])

    return images, labels


def normalize(vector, n_frames, range_min=-1, range_max=1, std_norm=False):
    vector = vector.reshape((n_frames, NUM_LANDMARKS, 3))
    nose_tips = vector[:, NOSE_TIP_IDX]
    distances = vector - nose_tips.reshape(n_frames, 1, 3)
    if std_norm:
        st_devs = np.std(vector, axis=1, keepdims=True)
        return distances / st_devs

    # MinMax scaling
    frame_min = distances.min(axis=1, keepdims=True)
    frame_max = distances.max(axis=1, keepdims=True)
    normalized = (distances - frame_min) / (frame_max - frame_min)
    normalized = normalized * (range_max - range_min) + range_min

    return normalized


def process_path(file_path):
    # load the raw data from the file as a string
    vector = np.load(file_path).astype(np.float32)
    n_frames = int(len(vector) / NUM_LANDMARKS)  # 478 is the number of landmarks coordinates
    vector = normalize(vector, n_frames, STD_NORM)
    vector = vector.reshape((n_frames, NUM_LANDMARKS * 3))
    if n_frames > FRAMES_TOTAL:
        start_points = list(range(0, n_frames - FRAMES_TOTAL))  # possible starting points
        start_index = random.choice(start_points)
        vector = vector[start_index: start_index + FRAMES_TOTAL]
    elif n_frames < FRAMES_TOTAL:
        empty = np.zeros((FRAMES_TOTAL - n_frames, vector.shape[1])).astype(np.float32)
        vector = tf.concat([empty, vector], axis=0)  # put empty values at the beginning

    label = extract_label(file_path)
    labels = tf.constant(np.full(shape=[len(CLASSES)], fill_value=label))

    return vector, labels


def build_tf_dataset(vectors_dir):
    assert os.path.isdir(vectors_dir)
    paths = [os.path.join(vectors_dir, filename) for filename in os.listdir(vectors_dir)]
    tf_dataset = tf.data.Dataset.from_tensor_slices(paths)

    tf_dataset = tf_dataset.map(lambda path: tf.numpy_function(process_path, [path], [tf.float32, tf.int8]),
                                num_parallel_calls=AUTOTUNE)
    tf_dataset = tf_dataset.map(_fixup_shape)
    tf_dataset = tf_dataset.batch(BATCH_SIZE)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


if __name__ == "__main__":
    train_dir = 'D:\\app faccia emozioni\\lm\\train\\'
    val_dir = 'D:\\app faccia emozioni\\lm\\val\\'
    test_dir = 'D:\\app faccia emozioni\\lm\\test\\'

    train_dataset = build_tf_dataset(train_dir)
    val_dataset = build_tf_dataset(val_dir)
    test_dataset = build_tf_dataset(test_dir)

    for x, y in train_dataset.take(1):
        print(x.shape)
