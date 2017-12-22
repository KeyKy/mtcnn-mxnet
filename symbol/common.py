import mxnet as mx
from common import mtcnn_output


data_names = ['data', ]
label_names = ['prob_label', 'regr_label']


def conv(data, num_filter, kernel, stride, pad, name):
    convolution = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                                     stride=stride, pad=pad, name='conv_%s' % name)
    relu = mx.sym.LeakyReLU(data=convolution, act_type='prelu', name='relu_%s' % name)
    return relu
