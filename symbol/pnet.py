import mxnet as mx
from common import mtcnn_output


data_names = ['data', ]
label_names = ['prob_label', 'regr_label']


def conv(data, num_filter, kernel, stride, pad, name):
    convolution = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                                     stride=stride, pad=pad, name='conv_%s' % name)
    # relu = mx.sym.Activation(data=convolution, act_type='relu', name='relu_%s' % name)
    relu = mx.sym.LeakyReLU(data=convolution, act_type='prelu', name='relu_%s' % name)
    return relu


def get_symbol(is_train=True):
    data = mx.sym.Variable(name='data')

    conv0 = conv(data=data, num_filter=10, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='p0')
    pool0 = mx.sym.Pooling(data=conv0, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pooling_convention='full', pool_type='max', name='pool_p0')
    conv1 = conv(data=pool0, num_filter=16, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='p1')
    conv2 = conv(data=conv1, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='p2')

    conv3 = mx.sym.Convolution(data=conv2, num_filter=2, kernel=(1, 1), name='conv_p3')

    prob = mx.sym.SoftmaxActivation(data=conv3, mode='channel', name='prob_p')
    regr = mx.sym.Convolution(data=conv2, num_filter=4, kernel=(1, 1), name='regr_p')

    if is_train:
        prob_label = mx.sym.Variable(name='prob_label')
        regr_label = mx.sym.Variable(name='regr_label')

        prob_flat = mx.sym.flatten(data=prob)
        regr_flat = mx.sym.flatten(data=regr)

        return mx.sym.Custom(prob=prob_flat, regr=regr_flat,
                             prob_label=prob_label, regr_label=regr_label, name='mtcnn', op_type='MtcnnOutput')
    else:
        return mx.sym.Group([prob, regr])
