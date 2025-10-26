import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
import shutil
import tqdm
import glob

from ultralytics import YOLO

# load pre-trained model
detection_model = YOLO("yolov8n.pt")

# choose random image
# img = random.choice(os.listdir(images_path))

i=detection_model.predict(source='https://i.imgur.com/7DbG6Kt.jpeg', conf=0.5, save=True, line_thickness=2, hide_labels=False)

im = plt.imread('/kaggle/working/runs/detect/predict/GRdCC.jpg')
plt.figure(figsize=(20,10))
plt.axis('off')
plt.imshow(im)