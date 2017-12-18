from common.mtcnn_iter import MtcnnRecordIter
from dataset import wider_face as ds

input_size = 48
batch_size = 1024
img_cnt = ds.img_cnt


def get_common_params():
    param = dict()
    param['batch_size'] = batch_size
    param['input_size'] = input_size
    param['neg_per_face'] = ds.neg_per_face
    param['pos_per_face'] = ds.pos_per_face
    return param


def get_test_iter():
    return MtcnnRecordIter(
        img_root='/home/yetiancai/data/wider-face/images/',
        bbox_file='/home/yetiancai/data/wider-face/bbox.val',
        **get_common_params()
    )


def get_train_iter():
    return MtcnnRecordIter(
        img_root='/home/yetiancai/data/wider-face/images/',
        bbox_file='/home/yetiancai/data/wider-face/bbox.train',
        # max_random_contrast=0.2,
        # max_random_illumination=30,
        **get_common_params()
    )

