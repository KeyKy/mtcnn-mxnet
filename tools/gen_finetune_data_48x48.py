
import cv2
from collections import namedtuple
import numpy as np
import mxnet as mx
import random

Batch = namedtuple('Batch', ['data'])


batch_size = 1
input_size = 24


rnet_prefix = '/home/yetiancai/models/mtcnn_mine/rnet'


def build_model(model_prefix):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 25)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, input_size, input_size))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod


mod = build_model(rnet_prefix)


def process(mod, rec_prefix, write_rec_prefix):
    record = mx.recordio.MXIndexedRecordIO(rec_prefix + '.idx', rec_prefix + '.rec', 'r')
    write_record = mx.recordio.MXIndexedRecordIO(write_rec_prefix + '.idx', write_rec_prefix + '.rec', 'w')

    s_list = []
    pos_cnt = 0
    neg_cnt = 0

    processed_cnt = 0
    while True:
        s = record.read()
        if not s:
            break
        processed_cnt += 1
        if processed_cnt % 1000 == 0:
            print('%s processed.' % processed_cnt)

        header, img = mx.recordio.unpack_img(s)
        if header.label[0] < 0:
            continue
        img = cv2.resize(img, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, [2, 0, 1])
        img = img[np.newaxis, :]
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0

        mod.forward(Batch([mx.nd.array(img)]))
        res = mod.get_outputs()[0].asnumpy()
        prob = res[0][1]
        if prob > 0.5:
            s_list.append(s)
            if header.label[0] == 0:
                neg_cnt += 1
            else:
                pos_cnt += 1

            # header, img = mx.recordio.unpack_img(s)
            # cv2.namedWindow('hello')
            # cv2.imshow('hello', img)
            # cv2.waitKey()
        else:
            continue

    random.shuffle(s_list)
    for i, s in enumerate(s_list):
        write_record.write_idx(i, s)

    print('pos_cnt: %s, neg_cnt: %s' % (pos_cnt, neg_cnt))


process(mod, '/home/yetiancai/rec/idl-24x24-finetune/val', '/home/yetiancai/rec/idl-48x48-finetune/val')
process(mod, '/home/yetiancai/rec/idl-24x24-finetune/train', '/home/yetiancai/rec/idl-48x48-finetune/train')
