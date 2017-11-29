from common.mtcnn_iter import MtcnnRecordIter

# img_cnt = 3211000
img_cnt = 21084000
# img_cnt = 619000

input_size = 12
batch_size = 512
preprocess_threads = 22


def get_common_params():
    param = dict()
    param['mean_r'] = 127.5
    param['mean_g'] = 127.5
    param['mean_b'] = 127.5
    param['scale'] = 1.0 / 128.0
    param['data_shape'] = (3, input_size, input_size)
    param['label_width'] = 5
    param['batch_size'] = batch_size
    param['preprocess_threads'] = preprocess_threads
    param['resize'] = input_size
    param['input_size'] = input_size
    param['shuffle'] = True

    return param


def get_test_iter():
    return MtcnnRecordIter(
        path_imgrec='/home/yetiancai/rec/idl-12x12/val.rec',
        **get_common_params()
    )


def get_train_iter():
    return MtcnnRecordIter(
        rand_mirror=True,
        path_imgrec='/home/yetiancai/rec/idl-12x12/train.rec',
        # max_random_contrast=0.2,
        # max_random_illumination=20,
        **get_common_params()
    )

