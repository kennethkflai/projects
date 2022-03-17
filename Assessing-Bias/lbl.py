import cv2
from glob import glob
import bz2
import numpy as np
import os

from mtcnn import MTCNN

path = r'data\*\*f*.ppm'

ss = glob(path)

train_data=[]
train_label=[]
test_data=[]
test_label=[]
flag = True
detector = MTCNN()
for im_index in range(2676, len(ss)):

    path = ss[im_index]
    print(path)
    subject = path[path.find('\\')+1:path.rfind('\\')]
    img = cv2.imread(path)
    temp = detector.detect_faces(img)

    if len(temp) >0:
        x1, y1, width, height = temp[0]['box']
        x2, y2 = x1 + width, y1 + height
        x1 = max(0, x1)

        c_img = img[y1:y2, x1:x2]

        img = cv2.resize(c_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        sv_path = path[:-3] + 'png'
        cv2.imwrite(sv_path,img)