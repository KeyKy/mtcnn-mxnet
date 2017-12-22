import mxnet as mx
import hyper_params.onet as hp


def get_train_params(begin_epoch=0):
    """
    get the training params
    """
    epoch_size = hp.img_cnt / hp.batch_size

    return {
        'num_epoch': hp.num_epoch,
        'kvstore': mx.kvstore.create('local'),
        'begin_epoch': begin_epoch,
        'optimizer': 'sgd',
        'optimizer_params': {
            'begin_num_update': begin_epoch * epoch_size,
            'learning_rate': hp.learning_rate,
            'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(
                step=[x * epoch_size for x in hp.epoch_steps], factor=0.1
            ),
            'wd': 0.0001,
            'momentum': 0.9,
        },

        'initializer': mx.initializer.Xavier(
            rnd_type='uniform', factor_type='avg', magnitude=2.34,
        ),
    }

