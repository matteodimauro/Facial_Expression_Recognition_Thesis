import os
import numpy as np
import cv2
import random
import math
import mediapipe as mp
from tqdm import tqdm


# resizing dimensions for displaying image here on Colab
RESIZED_HEIGHT = 360
RESIZED_WIDTH = 360

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh


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


def data_creator(videos, save_dir):
    for video in tqdm(videos):
        filename = os.path.basename(video)
        x = extract_landmarks(video)
        np.save(os.path.join(save_dir, filename[:-4]), x)
    print('End preprocessing {}'.format(save_dir))

    return


if __name__ == "__main__":
    video_dir = r'C:\Users\madimauro\Desktop\Progetto_Face_Emotion_Recognition\Datasets\Clone\CREMA-D\VideoFlash'
    videopaths = [os.path.join(video_dir, filename) for filename in os.listdir(video_dir)]
    print('Video files detected: ' + str(len(videopaths)) + '\n')
    data = dict()
    data['training'], data['validation'], data['testing'] = split_videos(videopaths)

    output_dir = r"C:\Users\madimauro\Desktop\Progetto_Face_Emotion_Recognition\numpy_array"
    train_dir = os.path.join(output_dir, r"train\\")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, r"val\\")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, r"test\\")
    os.makedirs(test_dir, exist_ok=True)

    data_creator(data['training'],
                 save_dir=train_dir)
    data_creator(data['validation'],
                 save_dir=val_dir)
    data_creator(data['testing'],
                 save_dir=test_dir)

    print('Dataset building completed')
