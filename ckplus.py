import os
import numpy as np
import cv2
import random
import math
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import tensorflow as tf
import glob
from imgaug import augmenters as iaa

#################################################################################
# 'Emotion' FOLDER MUST BE IN THE SAME DIRECTORY OF 'cohn-kanade-images' FOLDER #
#################################################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh

# Resizing dimensions
RESIZED_HEIGHT = 360
RESIZED_WIDTH = 360

# Loading parameters
FRAMES_TOTAL = 30
NUM_LANDMARKS = 478
NOSE_TIP_IDX = 19
STD_NORM = False

# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
CLASSES = {
    "ANG": 1,
    "CON": 2,
    "DIS": 3,
    "FEA": 4,
    "HAP": 5,
    "NEU": 0,
    "SAD": 6,
    "SUR": 7
}
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 20


def split_videos(paths, total_subjects=None, train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.seed(seed)
    subjects = list({path for path in paths})
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


def extract_landmarks(frame_path):
    # Opens the Video file
    mesh_list = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        frame = cv2.imread(frame_path)  # works with single images files too
        failed = 0
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
            print('Could not detected landmarks in file {}'.format(frame_path))
            failed += 1
            print('Failed frames count {}'.format(failed))

        return np.array(mesh_list, dtype=float)


def extract_aug_landmark(frame_path):
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
        frame = cv2.imread(frame_path)
        failed = 0
        i = 0

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results2 = face_mesh.process(cv2.cvtColor(personal_flip(frame), cv2.COLOR_BGR2RGB))
        results3 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, -15), cv2.COLOR_BGR2RGB))
        results4 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, 10), cv2.COLOR_BGR2RGB))
        results5 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, 5), cv2.COLOR_BGR2RGB))

        try:
            for r, k in enumerate(results.multi_face_landmarks[0].landmark):
                mesh_list.append([k.x, k.y, k.z])
        except TypeError:
            pass
        try:
            for r, k in enumerate(results2.multi_face_landmarks[0].landmark):
                mesh_list2.append([k.x, k.y, k.z])
        except TypeError:
            pass
        try:
            for r, k in enumerate(results3.multi_face_landmarks[0].landmark):
                mesh_list3.append([k.x, k.y, k.z])
        except TypeError:
            pass
        try:
            for r, k in enumerate(results4.multi_face_landmarks[0].landmark):
                mesh_list4.append([k.x, k.y, k.z])
        except TypeError:
            pass
        try:
            for r, k in enumerate(results5.multi_face_landmarks[0].landmark):
                mesh_list5.append([k.x, k.y, k.z])
        except TypeError:
            pass

        i += 1


        # vector[0, :] = np.array(mesh_list, dtype=float)
        # vector[1, :] = np.array(mesh_list2, dtype=float)
        # vector[2, :] = np.array(mesh_list3, dtype=float)
        # vector[3, :] = np.array(mesh_list4, dtype=float)
        # vector[4, :] = np.array(mesh_list5, dtype=float)

        return np.array(mesh_list, dtype=float), np.array(mesh_list2, dtype=float), np.array(mesh_list3, dtype=float), \
               np.array(mesh_list4, dtype=float), np.array(mesh_list5, dtype=float)


def data_creator(subjects_root, save_dir):
    inv_classes = {v: k for k, v in CLASSES.items()}
    for images_dir in tqdm(subjects_root):
        moments = list({os.path.join(images_dir, path) for path in os.listdir(images_dir)
                        if '.DS_Store' not in path})
        try:
            for moment in moments:
                # check that label exist, otherwise skip frames
                split_path = moment.split(os.sep)
                split_path[-3] = 'Emotion'  # point to labels directory
                emotion_dir = os.sep.join(split_path)
                label = read_label_from_file(emotion_dir)
                if not label:
                    continue
                frames_ldmks = []
                frames_dir = list({os.path.join(moment, path) for path in os.listdir(moment)
                                   if '.DS_Store' not in path})
                for frame_path in frames_dir:
                    landmarks = extract_landmarks(frame_path)
                    frames_ldmks.append(landmarks)
                np.save(os.path.join(save_dir, split_path[-2] + '_' + split_path[-1] + '_' + inv_classes[label] + '_'),
                        np.array(frames_ldmks))
        except Exception:
            print('Skipped ', images_dir)
            continue
    print('End preprocessing {}'.format(save_dir))

    return


def data_creator_second(subjects_root, save_dir):
    inv_classes = {v: k for k, v in CLASSES.items()}
    for images_dir in tqdm(subjects_root):
        moments = list({os.path.join(images_dir, path) for path in os.listdir(images_dir)
                        if '.DS_Store' not in path})
        try:
            for moment in moments:
                # check that label exist, otherwise skip frames
                split_path = moment.split(os.sep)
                split_path[-3] = 'Emotion'  # point to labels directory
                emotion_dir = os.sep.join(split_path)
                label = read_label_from_file(emotion_dir)
                if not label:
                    continue
                frames_ldmks = lndmk2 = lndmk3 = lndmk4 = lndmk5 = []
                frames_dir = list({os.path.join(moment, path) for path in os.listdir(moment)
                                   if '.DS_Store' not in path})
                for frame_path in frames_dir:

                    x, y, z, w, k = extract_aug_landmark(frame_path)
                    frames_ldmks.append(x)
                    lndmk2.append(y)
                    lndmk3.append(z)
                    lndmk4.append(w)
                    lndmk5.append(k)

                np.save(os.path.join(save_dir, split_path[-2] + '_' + split_path[-1] + '_' + inv_classes[label] + '_'),
                        np.array(frames_ldmks))
                np.save(os.path.join(save_dir, 'Aug1' + split_path[-2] + '_' + split_path[-1] + '_' + inv_classes[label] + '_'),
                        np.array(frames_ldmks))
                np.save(os.path.join(save_dir, 'Aug2' + split_path[-2] + '_' + split_path[-1] + '_' + inv_classes[label] + '_'),
                        np.array(frames_ldmks))
                np.save(os.path.join(save_dir, 'Aug3' + split_path[-2] + '_' + split_path[-1] + '_' + inv_classes[label] + '_'),
                        np.array(frames_ldmks))
                np.save(os.path.join(save_dir, 'Aug4' + split_path[-2] + '_' + split_path[-1] + '_' + inv_classes[label] + '_'),
                        np.array(frames_ldmks))
        except Exception:
            print('Skipped ', images_dir)
            continue
    print('End preprocessing {}'.format(save_dir))

    return


def read_label_from_file(emotion_dir):
    # labels_dir = 'C:\\Users\\maturisell\\Progetti\\Tesi Matteo Di Mauro\\ckplus\\Emotion'
    # filename = os.path.basename(path)
    # file_splits = filename[:-4].split('_')
    # subject, emotion = file_splits[-2], file_splits[-1]
    # emotion_dir = os.path.join(labels_dir, subject, emotion)
    emotion_pattern = glob.glob(os.path.join(emotion_dir, '*_emotion.txt'))
    if emotion_pattern:
        emotion_file = emotion_pattern[0]
        with open(emotion_file, 'r') as f:
            label = int(f.read().strip()[0])
            return label
    else:
        return None


def extract_label(path):
    parts = tf.strings.split(path, os.path.sep)
    one_hot = tf.strings.split(parts[-1], '_')[-2] == tf.constant(list(CLASSES.keys()))

    return tf.cast(one_hot, tf.int8)


def _fixup_shape(images, labels):
    images.set_shape([FRAMES_TOTAL, NUM_LANDMARKS * 3])
    labels.set_shape([len(CLASSES)])

    return images, labels


# function made on purpose to achieve a basic normalization
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
    n_frames = vector.shape[0]
    vector = basic_normalize(vector, n_frames)
    # vector = vector.reshape((n_frames * NUM_LANDMARKS, 3))
    # vector = basic_normalize(vector, n_frames)
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
    subjects_dir = 'D:\\DATA\\Cohn-Kanade Database\\Cohn-Kanade Database\\CK+\\cohn-kanade-images'
    subject_paths = [os.path.join(subjects_dir, filename) for filename in os.listdir(subjects_dir)]
    print('Subjects directories detected: ' + str(len(subject_paths)) + '\n')
    data = dict()
    data['training'], data['validation'], data['testing'] = split_videos(subject_paths)

    output_dir = 'D:\\app faccia emozioni\\augmented-ck+'
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

    # train_dir = 'C:\\Users\\maturisell\\Progetti\\Tesi Matteo Di Mauro\\ckplus-prep\\train'
    # val_dir = 'C:\\Users\\maturisell\\Progetti\\Tesi Matteo Di Mauro\\ckplus-prep\\val'
    # test_dir = 'C:\\Users\\maturisell\\Progetti\\Tesi Matteo Di Mauro\\ckplus-prep\\test'
    #
    # train_dataset = build_tf_dataset(train_dir)
    # val_dataset = build_tf_dataset(val_dir)
    # test_dataset = build_tf_dataset(test_dir)
    #
    # for x, y in train_dataset.take(10):
    #     print(x.shape, y.shape)
