import os
import numpy as np
import matplotlib.pyplot as plt


# gt_path = '/data/idl/annotations/bbox.val'
# pred_path = '/data/idl/annotations/bbox.eval'
gt_path = '/data/idl/annotations/bbox_nose_24.test'
pred_path = '/data/idl/annotations/bbox_nose_24.eval'
# gt_path = 'E:/wider-face/bbox.val'
# pred_path = 'E:/wider-face/bbox.eval'


def IoU(box, boxes):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


with open(gt_path) as f:
    lines = f.readlines()

gt = dict()
for line in lines:
    record = line.strip().split()
    fname = record[0]
    bboxes = np.array([float(v) for v in record[1:]]).reshape((-1, 4))
    gt[fname] = bboxes


with open(pred_path) as f:
    lines = f.readlines()


pred = dict()
for line in lines:
    record = line.strip().split()
    fname = record[0]
    bboxes = np.array([float(v) for v in record[1:]]).reshape((-1, 5))
    pred[fname] = bboxes


sum_faces = 0
for fname, bboxes in gt.items():
    sum_faces += len(bboxes)


res = []
for fname, bboxes_info in pred.items():
    if fname not in gt:
        continue
    bboxes = gt[fname]
    if len(bboxes) == 0:
        continue
    for bbox_info in bboxes_info:
        score = bbox_info[0]
        bbox = bbox_info[1:]
        iou = IoU(bbox, bboxes)
        if np.max(iou) > 0.65:
            res.append((score, 1))
        else:
            res.append((score, 0))


res = sorted(res, key=lambda i: i[0], reverse=True)

res_pts = []
sum_hit = 0
sum_not_hit = 0

flag = 0
for i, (score, hit) in enumerate(res):
    if hit:
        sum_hit += 1
    else:
        sum_not_hit += 1
    if sum_not_hit > 8000:
        print(res_pts[-1][-1])
        break
    res_pts.append((sum_not_hit, sum_hit / sum_faces))

    if sum_not_hit == 880:
        print('ff: ', sum_hit / sum_faces)

    if sum_hit / sum_faces > 0.85 and not flag:
        flag = 1
        print('sum not hit: ', sum_not_hit)


plt.figure()
plt.ylim((0.0, 1.0))
plt.xlabel('false positive')
plt.ylabel('precision')

plt.yticks([x*0.1 for x in range(0, 10)])

plt.plot([p[0] for p in res_pts], [p[1] for p in res_pts])
plt.show()


sum_pics = 0
for fname, bboxes in gt.items():
    if len(bboxes):
        sum_pics += 1

print('sum pics: %d' % sum_pics)

