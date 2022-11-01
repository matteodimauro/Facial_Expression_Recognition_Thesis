import os
import numpy as np
import cv2
import random
import math
import mediapipe as mp
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import albumentations as A
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from scipy.ndimage.interpolation import rotate
from imgaug import augmenters as iaa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh

# Resizing dimensions
RESIZED_HEIGHT = 360
RESIZED_WIDTH = 360

# Loading parameters
FRAMES_TOTAL = 85
NUM_LANDMARKS = 478
NOSE_TIP_IDX = 19
STD_NORM = True
CLASSES = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64


def vis_keypoints(image, keypoints, diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    image = cv2.resize(image, (500, 500))
    cv2.imshow('vis_jeypoints', image)


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


def split_videos(paths, total_subjects=None, train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.seed(seed)
    subjects = list({os.path.basename(path)[:4] for path in paths})
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
            transform = A.Compose([
                A.CenterCrop(width=360, height=360),
            ])
            transformed = transform(image=frame)
            frame = transformed['image']

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


def data_creator(videos, save_dir):
    for video in tqdm(videos):
        filename = os.path.basename(video)
        x = extract_landmarks(video)
        np.save(os.path.join(save_dir, filename[:-4]), x)
    print('End preprocessing {}'.format(save_dir))

    return


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

    return normalized


def process_path(file_path):
    # load the raw data from the file as a string
    vector = np.load(file_path).astype(np.float32)
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

    label = extract_label(file_path)
    labels = tf.constant(np.full(shape=[len(CLASSES)], fill_value=label))

    return vector, labels


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
    vector = np.zeros(shape=(5, FRAMES_TOTAL, NUM_LANDMARKS * 3)) # originally dimension 14

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
            #######################################################################################
            # SELECT (COMMENT/UNCOMMENT) HOW MUCH, AND WHICH, AUGMENTATION TO USE FOR THE DATASET #
            #######################################################################################

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results2 = face_mesh.process(cv2.cvtColor(personal_flip(frame), cv2.COLOR_BGR2RGB))
            results3 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, -15), cv2.COLOR_BGR2RGB))
            results4 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, 10), cv2.COLOR_BGR2RGB)) # originally -10
            results5 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, 5), cv2.COLOR_BGR2RGB)) # originally rotate -15
            # results6 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, 5), cv2.COLOR_BGR2RGB))
            # results7 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, 10), cv2.COLOR_BGR2RGB))
            # results8 = face_mesh.process(cv2.cvtColor(personal_rotate(frame, 15), cv2.COLOR_BGR2RGB))
            # results9 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, -15), cv2.COLOR_BGR2RGB))
            # results10 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, -10), cv2.COLOR_BGR2RGB))
            # results11 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, -5), cv2.COLOR_BGR2RGB))
            # results12 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, 5), cv2.COLOR_BGR2RGB))
            # results13 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, 10), cv2.COLOR_BGR2RGB))
            # results14 = face_mesh.process(cv2.cvtColor(personal_flip_2(frame, 15), cv2.COLOR_BGR2RGB))

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
            # try:
            #     for r, k in enumerate(results6.multi_face_landmarks[0].landmark):
            #         mesh_list6.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue
            # try:
            #     for r, k in enumerate(results7.multi_face_landmarks[0].landmark):
            #         mesh_list7.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue
            # try:
            #     for r, k in enumerate(results8.multi_face_landmarks[0].landmark):
            #         mesh_list8.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue
            # try:
            #     for r, k in enumerate(results9.multi_face_landmarks[0].landmark):
            #         mesh_list9.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue
            # try:
            #     for r, k in enumerate(results10.multi_face_landmarks[0].landmark):
            #         mesh_list10.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue
            # try:
            #     for r, k in enumerate(results11.multi_face_landmarks[0].landmark):
            #         mesh_list11.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue
            # try:
            #     for r, k in enumerate(results12.multi_face_landmarks[0].landmark):
            #         mesh_list12.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue
            # try:
            #     for r, k in enumerate(results13.multi_face_landmarks[0].landmark):
            #         mesh_list13.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue
            # try:
            #     for r, k in enumerate(results14.multi_face_landmarks[0].landmark):
            #         mesh_list14.append([k.x, k.y, k.z])
            # except TypeError:
            #     continue

            i += 1

        cap.release()

        # vector[0, :] = np.array(mesh_list, dtype=float)
        # vector[1, :] = np.array(mesh_list2, dtype=float)
        # vector[2, :] = np.array(mesh_list3, dtype=float)
        # vector[3, :] = np.array(mesh_list4, dtype=float) # np vector can't have fixed version in general
        # vector[4, :] = np.array(mesh_list5, dtype=float)
        # vector[5, :] = adapter(mesh_list6) # adapter was for a live version of extraction method
        # vector[6, :] = adapter(mesh_list6)
        # vector[7, :] = adapter(mesh_list8)
        # vector[8, :] = adapter(mesh_list9)
        # vector[9, :] = adapter(mesh_list10)
        # vector[10, :] = adapter(mesh_list11)
        # vector[11, :] = adapter(mesh_list12)
        # vector[12, :] = adapter(mesh_list13)
        # vector[13, :] = adapter(mesh_list14)

    return np.array(mesh_list, dtype=float), np.array(mesh_list2, dtype=float), np.array(mesh_list3, dtype=float), \
           np.array(mesh_list4, dtype=float), np.array(mesh_list5, dtype=float)


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


def build_tf_dataset_live(videos_dir):
    tf_dataset = tf.data.Dataset.list_files(videos_dir)

    tf_dataset = tf_dataset.map(lambda path: tf.numpy_function(extract_aug_landmark(), [path], [tf.float32, tf.int8]),
                                num_parallel_calls=AUTOTUNE)
    tf_dataset = tf_dataset.map(_fixup_shape)
    # for x, y in tf_dataset.take(1):
    #     print(x.shape)
    tf_dataset = tf_dataset.batch(BATCH_SIZE)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


def data_creator_second(videos, save_dir):
    for video in tqdm(videos):
        filename = os.path.basename(video)
        x, y, z, w, k = extract_aug_landmark(video)
        np.save(os.path.join(save_dir, filename[:-4]), x)
        np.save(os.path.join(save_dir, 'Aug1'+filename[:-4]), y)
        np.save(os.path.join(save_dir, 'Aug2'+filename[:-4]), z)
        np.save(os.path.join(save_dir, 'Aug3'+filename[:-4]), w)
        np.save(os.path.join(save_dir, 'Aug4'+filename[:-4]), k)
        # np.save(os.path.join(save_dir, 'Aug5'+filename[:-4]), x[5])
        # np.save(os.path.join(save_dir, 'Aug6'+filename[:-4]), x[6])
        # np.save(os.path.join(save_dir, 'Aug7'+filename[:-4]), x[7])
        # np.save(os.path.join(save_dir, 'Aug8'+filename[:-4]), x[8])
        # np.save(os.path.join(save_dir, 'Aug9'+filename[:-4]), x[9])
        # np.save(os.path.join(save_dir, 'Aug10'+filename[:-4]), x[10])
        # np.save(os.path.join(save_dir, 'Aug11'+filename[:-4]), x[11])
        # np.save(os.path.join(save_dir, 'Aug12'+filename[:-4]), x[12])
        # np.save(os.path.join(save_dir, 'Aug13'+filename[:-4]), x[13])

    print('End preprocessing {}'.format(save_dir))

    return


if __name__ == "__main__":
    images_dir = r'D:\DATA\CREMA-D\VideoFlash'
    subject_paths = [os.path.join(images_dir, filename) for filename in os.listdir(images_dir)]
    print('Video files detected: ' + str(len(subject_paths)) + '\n')
    data = dict()
    data['training'], data['validation'], data['testing'] = split_videos(subject_paths)

    output_dir = r"D:\app faccia emozioni\augmented-cremad"
    train_dir = os.path.join(output_dir, r"train\\")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, r"val\\")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, r"test\\")
    os.makedirs(test_dir, exist_ok=True)

    # train_dataset = build_tf_dataset_live(data['training'])
    # for x, y in train_dataset.take(1):
    #     print(x.shape)

    # build_tf_dataset_live(data['training'])
    # build_tf_dataset_live(data['validation'])
    # build_tf_dataset_live(data['testing'])

    data_creator_second(data['training'],
                 save_dir=train_dir)
    data_creator_second(data['validation'],
                 save_dir=val_dir)
    data_creator_second(data['testing'],
                 save_dir=test_dir)

    print('Dataset building completed')

    # train_dir = 'C:\\Users\\madimauro\\Desktop\\Progetto_Face_Emotion_Recognition\\numpy_array\\train\\'
    # val_dir = 'C:\\Users\\madimauro\\Desktop\\Progetto_Face_Emotion_Recognition\\numpy_array\\val\\'
    # test_dir = 'C:\\Users\\madimauro\\Desktop\\Progetto_Face_Emotion_Recognition\\numpy_array\\test\\'
    #
    # train_dataset = build_tf_dataset(train_dir)
    # val_dataset = build_tf_dataset(val_dir)
    # test_dataset = build_tf_dataset(test_dir)
    #
    # for x, y in train_dataset.take(1):
    #     print(x.shape)
