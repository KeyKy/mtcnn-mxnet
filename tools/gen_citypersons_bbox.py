import scipy.io as sio
import os

data = sio.loadmat('/data/CityPersons/ann/annotations/anno_val.mat')
data = data['anno_val_aligned'][0]

f = open('/data/CityPersons/ann/annotations/bbox.val', 'w')

for record in data:
    im_name = record['im_name'][0][0][0]
    bboxes = record['bbs'][0][0]

    im_name = os.path.join('val', im_name.split('_', 1)[0], im_name)

    resline = [im_name, ]
    for bbox in bboxes:
        class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis = bbox
        if class_label != 1:
            continue

        resline.append('%.2f %.2f %.2f %.2f' % (x1, y1, x1 + w, y1 + h))

    f.write(' '.join(resline) + '\n')

data = sio.loadmat('/data/CityPersons/ann/annotations/anno_train.mat')
data = data['anno_train_aligned'][0]

f = open('/data/CityPersons/ann/annotations/bbox.train', 'w')

for record in data:
    im_name = record['im_name'][0][0][0]
    bboxes = record['bbs'][0][0]

    im_name = os.path.join('train', im_name.split('_', 1)[0], im_name)

    resline = [im_name, ]
    for bbox in bboxes:
        class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis = bbox
        if class_label != 1:
            continue

        resline.append('%.2f %.2f %.2f %.2f' % (x1, y1, x1 + w, y1 + h))

    f.write(' '.join(resline) + '\n')
