import mxnet as mx
import dataset.pnet

gpus = (0, 1)
epoch_size = dataset.pnet.img_cnt / dataset.pnet.batch_size


def get_train_params(begin_epoch=0):
    """
    get the training params
    """
    return {
        'num_epoch': 20,
        'kvstore': mx.kvstore.create('local'),

        'begin_epoch': begin_epoch,
        'optimizer': 'sgd',
        'optimizer_params': {
            'begin_num_update': begin_epoch * epoch_size,
            'learning_rate': 0.01,
            # 'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(
            #     step=[x * epoch_size for x in (5, 10, 15)], factor=0.1
            # ),
            'wd': 0.0005,
            'momentum': 0.9,
        },

        'initializer': mx.initializer.Xavier(
            rnd_type='uniform', factor_type='avg', magnitude=2.34,
        ),
    }
