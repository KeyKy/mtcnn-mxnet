import sys
import mxnet as mx
from common import mtcnn_output


data_names = ['data', ]
label_names = ['prob_label', 'regr_label', 'last_stage']


def conv(data, num_filter, kernel, stride, pad, name):
    convolution = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                                     stride=stride, pad=pad, name='conv_%s' % name)
    # relu = mx.sym.Activation(data=convolution, act_type='relu', name='relu_%s' % name)
    relu = mx.sym.LeakyReLU(data=convolution, act_type='prelu', name='relu_%s' % name)
    return relu


def get_symbol(is_train=True):
    data = mx.sym.Variable(name='data')

    conv0 = conv(data=data, num_filter=28, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='r0')
    pool0 = mx.sym.Pooling(data=conv0, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pooling_convention='full', pool_type='max', name='pool_r0')

    conv1 = conv(data=pool0, num_filter=48, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='r1')
    pool1 = mx.sym.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pooling_convention='full', pool_type='max', name='pool_r1')

    conv2 = conv(data=pool1, num_filter=64, kernel=(2, 2), stride=(1, 1), pad=(0, 0), name='r2')

    fc = mx.sym.FullyConnected(data=conv2, num_hidden=128, name='fc_r')
    # act = mx.sym.Activation(data=fc, act_type='relu', name='relu_%s' % 'fc_r')
    act = mx.sym.LeakyReLU(data=fc, act_type='prelu', name='relu_%s' % 'fc_r')

    fc_prob = mx.sym.FullyConnected(data=act, num_hidden=2, name='fc_prob_r')
    prob = mx.sym.SoftmaxActivation(data=fc_prob, name='prob_r')

    regr = mx.sym.FullyConnected(data=act, num_hidden=4, name='regr_r')

    if is_train:
        prob_label = mx.sym.Variable(name='prob_label')
        regr_label = mx.sym.Variable(name='regr_label')
        last_stage = mx.sym.Variable(name='last_stage')

        return mx.sym.Custom(
            prob=prob, regr=regr,
            prob_label=prob_label, regr_label=regr_label, last_stage=last_stage,
            name='mtcnn', op_type='MtcnnOutput'
        )
    else:
        return mx.sym.Group([prob, regr])
