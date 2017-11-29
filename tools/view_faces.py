import os
import numpy as np
import cv2

img_root = '/home/yetiancai/data/idl/images'
train_bboxes = '/home/yetiancai/data/idl/annotations/bbox.train'
# img_root = '/home/yetiancai/data/wider-face/images/'
# train_bboxes = '/home/yetiancai/data/wider-face/bbox.train'


with open(train_bboxes, 'r') as f:
    lines = f.readlines()


for line in lines:
    record = line.strip().split()
    fname = record[0]

    if 'AFLW' not in fname:
        continue

    bboxes = np.array([int(float(v)) for v in record[1:]]).reshape((-1, 4))
    img = cv2.imread(os.path.join(img_root, fname))
    if img is None:
        continue
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))

    cv2.imshow('hello', img)
    cv2.waitKey()

