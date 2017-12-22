import mxnet as mx
from symbol.common import conv
import hyper_params.rnet as hp


def get_symbol(is_train=True):
    data = mx.sym.Variable(name='data')

    conv0 = conv(data=data, num_filter=28, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='r0')
    pool0 = mx.sym.Pooling(data=conv0, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pooling_convention='full', pool_type='max', name='pool_r0')

    conv1 = conv(data=pool0, num_filter=48, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='r1')
    pool1 = mx.sym.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pooling_convention='full', pool_type='max', name='pool_r1')

    conv2 = conv(data=pool1, num_filter=64, kernel=(2, 2), stride=(1, 1), pad=(0, 0), name='r2')

    fc = mx.sym.FullyConnected(data=conv2, num_hidden=128, name='fc_r')
    act = mx.sym.LeakyReLU(data=fc, act_type='prelu', name='relu_%s' % 'fc_r')

    fc_prob = mx.sym.FullyConnected(data=act, num_hidden=2, name='fc_prob_r')
    prob = mx.sym.SoftmaxActivation(data=fc_prob, name='prob_r')

    regr = mx.sym.FullyConnected(data=act, num_hidden=4, name='regr_r')

    if is_train:
        prob_label = mx.sym.Variable(name='prob_label')
        regr_label = mx.sym.Variable(name='regr_label')

        return mx.sym.Custom(
            prob=prob, regr=regr,
            prob_label=prob_label, regr_label=regr_label,
            focal_gamma=hp.focal_gamma, regr_weight=hp.regr_weight,
            name='mtcnn', op_type='MtcnnOutput'
        )
    else:
        return mx.sym.Group([prob, regr])
