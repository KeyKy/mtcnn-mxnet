import mxnet as mx
import numpy as np
import cv2
import os
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])


def detect(img, scale):
    h, w, c = img.shape

    sh = int(scale*h)
    sw = int(scale*w)

    simg = cv2.resize(img, (sw, sh))

    sym, arg_params, aux_params = mx.model.load_checkpoint('/home/yetiancai/models/mtcnn_test/det1', 1)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, sh, sw))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
    simg = np.transpose(simg, [2, 0, 1])
    simg = simg[np.newaxis, :]

    simg = simg.astype(np.float32)
    simg = (simg - 127.5) / 128.0

    mod.forward(Batch([mx.nd.array(simg)]))
    res = mod.get_outputs()[1].asnumpy()

    res = res[0][1] > 0.7

    bboxes = []

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if res[i][j]:
                x1 = j * 2 / scale
                y1 = i * 2 / scale
                x2 = (j * 2 + 12) / scale
                y2 = (i * 2 + 12) / scale
                bboxes.append((x1, y1, x2, y2))
    return bboxes


# for i in range(0, 5):
    # fname = '%d.jpg' % i
fname = 'img_495.jpg'
if not os.path.exists(fname):
    pass

img = cv2.imread(os.path.join('/home/yetiancai/workspace/mtcnn-mxnet/test/', fname))
print(img.shape)
bboxes = detect(img, 0.2)

for bbox in bboxes:
    bbox = [int(p) for p in bbox]
    print(bbox)
    cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255))
    cv2.imshow('hello', img)
    cv2.waitKey()


