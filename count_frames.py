import cv2
import numpy as np
import os

VIDEO_DIR = 'C:\\Users\\madimauro\\Desktop\\Progetto_Face_Emotion_Recognition\\Datasets\\Clone\\CREMA-D\VideoFlash\\'

# @title
videopaths = [os.path.join(VIDEO_DIR, filename) for filename in os.listdir(VIDEO_DIR)]
print(len(videopaths))
lengths = np.zeros(len(videopaths))
for i,p in enumerate(videopaths):
  cap = cv2.VideoCapture(p)
  length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  # print(i, length)
  lengths[i] = length

print(lengths.mean(), lengths.max())
print(len(lengths[lengths > 120]))
