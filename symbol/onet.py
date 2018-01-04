import mxnet as mx
from symbol.common import conv
import hyper_params.onet as hp
from hyper_params import lmk_cnt


def get_symbol(is_train=True):
    data = mx.sym.Variable(name='data')

    conv0 = conv(data=data, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='o0')
    pool0 = mx.sym.Pooling(data=conv0, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pooling_convention='full', pool_type='max', name='pool_o0')

    conv1 = conv(data=pool0, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='o1')
    pool1 = mx.sym.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pooling_convention='full', pool_type='max', name='pool_o1')

    conv2 = conv(data=pool1, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name='o2')
    pool2 = mx.sym.Pooling(data=conv2, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pooling_convention='full', pool_type='max', name='pool_o2')

    conv3 = conv(data=pool2, num_filter=128, kernel=(2, 2), stride=(1, 1), pad=(0, 0), name='o3')

    fc = mx.sym.FullyConnected(data=conv3, num_hidden=256, name='fc_o')
    # act = mx.sym.Activation(data=fc, act_type='relu', name='relu_%s' % 'fc_o')
    act = mx.sym.LeakyReLU(data=fc, act_type='prelu', name='relu_%s' % 'fc_o')

    fc_prob = mx.sym.FullyConnected(data=act, num_hidden=2, name='fc_prob_o')
    prob = mx.sym.SoftmaxActivation(data=fc_prob, name='prob_o')

    regr = mx.sym.FullyConnected(data=act, num_hidden=4, name='regr_o')

    lmks = mx.sym.FullyConnected(data=act, num_hidden=lmk_cnt * 2, name='lmks_o')

    if is_train:
        prob_label = mx.sym.Variable(name='prob_label')
        regr_label = mx.sym.Variable(name='regr_label')
        lmks_label = mx.sym.Variable(name='lmks_label')

        return mx.sym.Custom(
            prob=prob, regr=regr, lmks=lmks,
            prob_label=prob_label, regr_label=regr_label, lmks_label=lmks_label,
            focal_gamma=hp.focal_gamma, regr_weight=hp.regr_weight, lmks_weight=hp.lmks_weight,
            name='mtcnn', op_type='MtcnnOutput'
        )
    else:
        return mx.sym.Group([prob, regr, lmks])
