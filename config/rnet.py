import mxnet as mx
import dataset.rnet

gpus = (0, 1)
epoch_size = dataset.rnet.img_cnt / dataset.rnet.batch_size


def get_train_params(begin_epoch=0):
    """
    get the training params
    """
    return {
        'num_epoch': 25,
        'kvstore': mx.kvstore.create('local'),

        'begin_epoch': begin_epoch,
        'optimizer': 'sgd',
        'optimizer_params': {
            'begin_num_update': begin_epoch * epoch_size,
            'learning_rate': 0.05,
            'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(
                step=[x * epoch_size for x in (5, 10, 15, 20)], factor=0.1
            ),
            'wd': 0.0001,
            'momentum': 0.9,
        },

        'initializer': mx.initializer.Xavier(
            rnd_type='uniform', factor_type='avg', magnitude=2.34,
        ),
    }
