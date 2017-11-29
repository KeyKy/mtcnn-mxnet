
import os
import cv2
import random
import numpy as np
import numpy.random as npr


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


def gen_neg_data(img, bboxes, img_idx, save_dir):
    save_neg_dir = os.path.join(save_dir, 'neg')
    height, width, _ = img.shape
    neg_num = 0

    while neg_num < 10:
        size = npr.randint(20, min(width, height)/3.0)
        nx1 = npr.randint(0, width-size)
        ny1 = npr.randint(0, height-size)
        nx2 = nx1 + size
        ny2 = ny1 + size

        box = np.array([nx1, ny1, nx2, ny2])
        iou = IoU(box, bboxes)
        if np.max(iou) > 0.3:
            continue

        crop = img[ny1:ny2, nx1:nx2, :]
        save_path = os.path.join(save_neg_dir, '%05d-%03d.jpg' % (img_idx, neg_num))
        cv2.imwrite(save_path, cv2.resize(crop, (save_img_size, save_img_size), interpolation=cv2.INTER_LINEAR))
        neg_num += 1


def gen_pos_and_part(img, bboxes, img_idx, save_dir):
    save_pos_dir = os.path.join(save_dir, 'pos')
    save_part_dir = os.path.join(save_dir, 'part')

    pos_num = 0
    part_num = 0

    img_h, img_w, img_c = img.shape

    regr_infos = dict()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if max(w, h) < 20 or x1 < 0 or x2 < 0 or w < 0 or h < 0:
            continue

        side = max(w, h)
        sida = min(w, h)

        for i in range(5):
            if i > 0:
                size = npr.randint(int(sida * 0.8), np.ceil(side * 1.25))
                delta_x = npr.randint(-0.2*w, 0.2*w)
                delta_y = npr.randint(-0.2*h, 0.2*h)
            else:
                size = int(side)
                delta_x = 0
                delta_y = 0

            crop_x1 = int(max(x1 + w / 2.0 + delta_x - size / 2.0, 0))
            crop_y1 = int(max(y1 + h / 2.0 + delta_y - size / 2.0, 0))
            crop_x2 = crop_x1 + size
            crop_y2 = crop_y1 + size
            if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > img_w or crop_y2 > img_h:
                continue
            crop_bbox = np.array([crop_x1, crop_y1, crop_x2, crop_y2])

            iou = IoU(crop_bbox, bbox.reshape(1, -1))[0]
            if iou < 0.3:
                continue

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            regr_x1 = (x1 - crop_x1) / float(size)
            regr_y1 = (y1 - crop_y1) / float(size)
            regr_x2 = (x2 - crop_x2) / float(size)
            regr_y2 = (y2 - crop_y2) / float(size)

            if 0.6 < iou < 0.80:
                save_path = os.path.join(save_part_dir, '%05d-%03d.jpg' % (img_idx, part_num))
                part_num += 1
            elif 0.8 <= iou:
                save_path = os.path.join(save_pos_dir, '%05d-%03d.jpg' % (img_idx, pos_num))
                pos_num += 1
            else:
                continue

            cv2.imwrite(save_path, cv2.resize(crop, (save_img_size, save_img_size), interpolation=cv2.INTER_LINEAR))
            regr_infos[os.path.relpath(save_path, save_dir)] = np.array([regr_x1, regr_y1, regr_x2, regr_y2])

    return regr_infos


def gen_data(img_root, lines, save_dir):
    save_neg_dir = os.path.join(save_dir, 'neg')
    save_pos_dir = os.path.join(save_dir, 'pos')
    save_part_dir = os.path.join(save_dir, 'part')

    for dir_path in (save_neg_dir, save_pos_dir, save_part_dir):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    regr_infos = dict()
    for img_idx, line in enumerate(lines):
        if img_idx % 100 == 0:
            print('%d of %d img processed' % (img_idx, len(lines)))

        record = line.strip().split()
        img_path = os.path.join(img_root, record[0])
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, c = img.shape
        if min(h, w) < 40:
            continue

        bboxes = np.array([float(v) for v in record[1:]]).reshape((-1, 4))
        if bboxes.shape[0] < 1:
            continue

        gen_neg_data(img, bboxes, img_idx, save_dir)
        regr_infos.update(
            gen_pos_and_part(img, bboxes, img_idx, save_dir)
        )

    train_lst = []

    for filename in os.listdir(save_neg_dir):
        train_lst.append((os.path.join('neg', filename), [0, 0, 0, 0, 0]))

    for filename in os.listdir(save_pos_dir):
        filepath = os.path.join('pos', filename)
        if filepath not in regr_infos:
            print('1, error: ', filename, 'not in regr_infos')
            continue
        regr = regr_infos[filepath]
        train_lst.append((filepath, [1, regr[0], regr[1], regr[2], regr[3]]))

    for filename in os.listdir(save_part_dir):
        filepath = os.path.join('part', filename)
        if filepath not in regr_infos:
            print('2, error: ', filename, 'not in regr_infos')
            continue
        regr = regr_infos[filepath]
        train_lst.append((filepath, [-1, regr[0], regr[1], regr[2], regr[3]]))

    final_lst = [(i, content) for i, content in enumerate(train_lst)]
    random.shuffle(final_lst)
    with open(os.path.join(save_dir, 'train.lst'), 'w') as f:
        for i, content in final_lst:
            line = []
            line.append(str(i))
            for v in content[-1]:
                line.append('%.2f' % v)
            line.append(content[0])
            f.write('\t'.join(line) + '\n')


if __name__ == '__main__':
    idl_root = '/home/yetiancai/data/idl'
    save_img_size = 48

    for tname in ('val', 'train'):
        img_root = os.path.join(idl_root, 'images')
        bbox_file = os.path.join(idl_root, 'annotations/bbox.%s' % tname)
        save_dir = os.path.join(idl_root, '%s_crops' % tname)

        with open(bbox_file) as f:
            lines = f.readlines()
        random.shuffle(lines)
        gen_data(img_root, lines, save_dir)


# if __name__ == '__main__':
#     wider_root = '/home/yetiancai/data/wider-face'
#     wider_root = 'E:/wider-face'
#     for tname in ('val', 'train'):
#         img_root = os.path.join(wider_root, '%s/images' % tname)
#         bbox_file = os.path.join(wider_root, 'bbox.%s' % tname)
#         save_dir = os.path.join(wider_root, '%s/crops' % tname)
#         with open(bbox_file) as f:
#             lines = f.readlines()
#         random.shuffle(lines)
#         gen_data(img_root, lines, save_dir)

