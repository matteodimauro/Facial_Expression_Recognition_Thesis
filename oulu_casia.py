import os
import numpy as np
import cv2
import random
import math
import mediapipe as mp
from tqdm import tqdm
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from imgaug import augmenters as iaa

# resizing dimensions for displaying image here on Colab
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RESIZED_HEIGHT = 360
RESIZED_WIDTH = 360

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh

FRAMES_TOTAL = 85
NUM_LANDMARKS = 478
NOSE_TIP_IDX = 19
STD_NORM = True

CLASSES = {"A": 0,
           "D": 1,
           "F": 2,
           "H": 3,
           "S2": 4,
           "S1": 5
           }

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 30


def split_videos(paths, total_subjects=None, train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.seed(seed)
    subjects = list({os.path.basename(path)[3:6] for path in paths})
    random.shuffle(subjects)
    if total_subjects:
        subjects = subjects[:total_subjects]
    num_subjects = len(subjects)
    num_train = round(num_subjects * train_ratio)
    num_valid = round(num_subjects * valid_ratio)
    train_subjects = subjects[:num_train]
    valid_subjects = subjects[num_train: num_train + num_valid]
    test_subjects = subjects[num_train + num_valid:]
    train_paths = [path for path in paths for sub in train_subjects if sub in path]
    valid_paths = [path for path in paths for sub in valid_subjects if sub in path]
    test_paths = [path for path in paths for sub in test_subjects if sub in path]

    return train_paths, valid_paths, test_paths


def extract_landmarks(video):
    # Opens the Video file
    mesh_list = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        cap = cv2.VideoCapture(video)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print(video, "number of frames: {}".format(video_length))
        i = 0
        failed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize image to square
            h, w = frame.shape[:2]
            if h < w:
                frame = cv2.resize(frame, (RESIZED_WIDTH, math.floor(h / (w / RESIZED_WIDTH))))
            else:
                frame = cv2.resize(frame, (math.floor(w / (h / RESIZED_HEIGHT)), RESIZED_HEIGHT))

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                for r, k in enumerate(results.multi_face_landmarks[0].landmark):
                    mesh_list.append([k.x, k.y, k.z])
            except TypeError:
                print('Could not detected landmarks in frame {} of file {}'.format(i, video))
                failed += 1
                print('Failed frames count {}'.format(failed))
            i += 1
        cap.release()

        return np.array(mesh_list, dtype=float)


def personal_rotate(image, angle):
    seq = iaa.Sequential([
        iaa.Affine(rotate=angle),
        iaa.AdditiveGaussianNoise(scale=(10, 60))
    ])

    images_aug = seq(image=image)
    return images_aug


def personal_flip(image):
    seq = iaa.Sequential([
        iaa.Fliplr(1.0),
        iaa.AdditiveGaussianNoise(scale=(10, 60))
    ])

    images_aug = seq(image=image)
    return images_aug


def personal_flip_2(image, angle):
    seq = iaa.Sequential([
        iaa.Affine(rotate=angle),
        iaa.Fliplr(1.0),
        iaa.AdditiveGaussianNoise(scale=(10, 60))
    ])

    images_aug = seq(image=image)
    return images_aug


def adapter(vec):
    vector = np.array(vec, dtype=float)
    # load the raw data from the file as a string
    n_frames = int(len(vector) / NUM_LANDMARKS)  # 478 is the number of landmarks coordinates
    vector = basic_normalize(vector, n_frames)
    vector = vector.reshape((n_frames, NUM_LANDMARKS * 3))
    if n_frames > FRAMES_TOTAL:
        start_points = list(range(0, n_frames - FRAMES_TOTAL))  # possible starting points
        start_index = random.choice(start_points)
        vector = vector[start_index: start_index + FRAMES_TOTAL]
    elif n_frames < FRAMES_TOTAL:
        empty = np.zeros((FRAMES_TOTAL - n_frames, vector.shape[1])).astype(np.float32)
        vector = tf.concat([empty, vector], axis=0)  # put empty values at the beginning
    return vector

    return vector


def extract_aug_landmark(video):
    # number of lists depends on number of augmentations
    mesh_list = mesh_list2 = mesh_list3 = mesh_list4 = mesh_list5 = []
    # mesh_list6 = mesh_list7 = mesh_list8 = \
    # mesh_list9 = mesh_list10 = mesh_list11 = mesh_list12 = mesh_list13 = mesh_list14 = []
    vector = np.zeros(shape=(5, FRAMES_TOTAL, NUM_LANDMARKS * 3))  # originally dimension 14

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        cap = cv2.VideoCapture(video)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print(video, "number of frames: {}".format(video_length))
        failed = 0
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results2 = face_mesh.process(cv2.cvtColor(personal_flip(frame), cv2.COLOR_BGR2RGB))
            results3 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, -15), cv2.COLOR_BGR2RGB))
            results4 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, 10), cv2.COLOR_BGR2RGB))
            results5 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, 5), cv2.COLOR_BGR2RGB))

            try:
                for r, k in enumerate(results.multi_face_landmarks[0].landmark):
                    mesh_list.append([k.x, k.y, k.z])
            except TypeError:
                continue
            try:
                for r, k in enumerate(results2.multi_face_landmarks[0].landmark):
                    mesh_list2.append([k.x, k.y, k.z])
            except TypeError:
                continue
            try:
                for r, k in enumerate(results3.multi_face_landmarks[0].landmark):
                    mesh_list3.append([k.x, k.y, k.z])
            except TypeError:
                continue
            try:
                for r, k in enumerate(results4.multi_face_landmarks[0].landmark):
                    mesh_list4.append([k.x, k.y, k.z])
            except TypeError:
                continue
            try:
                for r, k in enumerate(results5.multi_face_landmarks[0].landmark):
                    mesh_list5.append([k.x, k.y, k.z])
            except TypeError:
                continue

            i += 1

        cap.release()

        # vector[0, :] = np.array(mesh_list, dtype=float)
        # print(vector)
        # vector[1, :] = np.array(mesh_list2, dtype=float)
        # vector[2, :] = np.array(mesh_list3, dtype=float)
        # vector[3, :] = np.array(mesh_list4, dtype=float)
        # vector[4, :] = np.array(mesh_list5, dtype=float)

    return np.array(mesh_list, dtype=float), np.array(mesh_list2, dtype=float), np.array(mesh_list3, dtype=float),\
            np.array(mesh_list4, dtype=float), np.array(mesh_list5, dtype=float)


def data_creator(videos, save_dir):
    for video in tqdm(videos):
        filename = os.path.basename(video)
        x = extract_landmarks(video)
        np.save(os.path.join(save_dir, filename[:-4] + '_'), x)
    print('End preprocessing {}'.format(save_dir))

    return


def data_creator_second(videos, save_dir):
    for video in tqdm(videos):
        filename = os.path.basename(video)
        x, y, z, w, k = extract_aug_landmark(video)
        np.save(os.path.join(save_dir, filename[:-4] + '_'), x)
        np.save(os.path.join(save_dir, 'Aug1' + filename[:-4] + '_'), y)
        np.save(os.path.join(save_dir, 'Aug2' + filename[:-4] + '_'), z)
        np.save(os.path.join(save_dir, 'Aug3' + filename[:-4] + '_'), w)
        np.save(os.path.join(save_dir, 'Aug4' + filename[:-4] + '_'), k)

    print('End preprocessing {}'.format(save_dir))

    return


# function made on purpose to achieve a basic normalization


def extract_label(path):
    parts = tf.strings.split(path, os.path.sep)
    one_hot = tf.strings.split(parts[-1], '_')[-2] == tf.constant(list(CLASSES.keys()))

    return tf.cast(one_hot, tf.int8)


def _fixup_shape(images, labels):
    images.set_shape([FRAMES_TOTAL, NUM_LANDMARKS * 3])
    labels.set_shape([len(CLASSES)])

    return images, labels


def basic_normalize(data, frames):
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
    print(normalized.min(), np.min(normalized))

    return normalized


def process_path(file_path):
    # load the raw data from the file as a string
    vector = np.load(file_path).astype(np.float32)
    n_frames = int(len(vector) / NUM_LANDMARKS)  # 478 is the number of landmarks coordinates
    # vector = normalize(vector, n_frames, STD_NORM)
    vector = basic_normalize(vector, n_frames)
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
    video_dir = r'D:\DATA\Oulu CASIA\Original Video\NI'
    videopaths = [os.path.join(video_dir, filename) for filename in os.listdir(video_dir)]
    print('Video files detected: ' + str(len(videopaths)) + '\n')
    data = dict()
    data['training'], data['validation'], data['testing'] = split_videos(videopaths)

    output_dir = r"D:\app faccia emozioni\augmented-casia"
    train_dir = os.path.join(output_dir, r"train\\")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, r"val\\")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, r"test\\")
    os.makedirs(test_dir, exist_ok=True)

    data_creator_second(data['training'],
                 save_dir=train_dir)
    data_creator_second(data['validation'],
                 save_dir=val_dir)
    data_creator_second(data['testing'],
                 save_dir=test_dir)

    print('Dataset building completed')

    # train_dir = 'C:\\Users\\madimauro\\Desktop\\Progetto_Face_Emotion_Recognition\\landmarks-CASIA\\train\\'
    # val_dir = 'C:\\Users\\madimauro\\Desktop\\Progetto_Face_Emotion_Recognition\\landmarks-CASIA\\val\\'
    # test_dir = 'C:\\Users\\madimauro\\Desktop\\Progetto_Face_Emotion_Recognition\\landmarks-CASIA\\test\\'
    #
    # train_dataset = build_tf_dataset(train_dir)
    # val_dataset = build_tf_dataset(val_dir)
    # test_dataset = build_tf_dataset(test_dir)
    #
    # for x, y in train_dataset.take(1):
    #     print(x.shape)
