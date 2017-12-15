from common.mtcnn_iter import MtcnnRecordIter

origin_image_cnt = 12880

input_size = 24
neg_per_face = 1
pos_per_face = 1
img_per_batch = 512
batch_size = (neg_per_face + pos_per_face) * img_per_batch
round_cnt = 100

img_cnt = origin_image_cnt // img_per_batch * batch_size * round_cnt


def get_common_params():
    param = dict()
    param['batch_size'] = batch_size
    param['input_size'] = input_size
    param['neg_per_face'] = neg_per_face
    param['pos_per_face'] = pos_per_face
    param['img_per_batch'] = img_per_batch
    param['round_cnt'] = round_cnt
    return param


def get_test_iter():
    return MtcnnRecordIter(
        img_root='/home/yetiancai/data/wider-face/val/images/',
        bbox_file='/home/yetiancai/data/wider-face/bbox.val',
        **get_common_params()
    )


def get_train_iter():
    return MtcnnRecordIter(
        img_root='/home/yetiancai/data/wider-face/train/images/',
        bbox_file='/home/yetiancai/data/wider-face/bbox.train',
        **get_common_params()
    )

