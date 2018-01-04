from common.mtcnn_iter import MtcnnRecordIter
import hyper_params.onet as hp


def get_common_params():
    param = dict()
    param['batch_size'] = hp.batch_size
    param['input_size'] = hp.input_size
    param['neg_per_face'] = hp.neg_per_face
    param['pos_per_face'] = hp.pos_per_face
    param['iou_thresh'] = hp.iou_thresh
    param['img_root'] = hp.img_root
    param['filters'] = hp.filters
    return param


def get_test_iter():
    return MtcnnRecordIter(
        bbox_file=hp.bbox_val,
        lmks_img_root='/home/yetiancai/data/idl/images/',
        lmks_file='/home/yetiancai/data/idl/annotations/all.test',
        **get_common_params()
    )


def get_train_iter():
    return MtcnnRecordIter(
        bbox_file=hp.bbox_train,
        lmks_img_root='/home/yetiancai/data/idl/images/',
        lmks_file='/home/yetiancai/data/idl/annotations/all.train',
        **get_common_params()
    )

