import mxnet as mx
from hyper_params import lmk_cnt


class MtcnnOutput(mx.operator.CustomOp):
    def __init__(self, focal_gamma, regr_weight, lmks_weight):
        super(MtcnnOutput, self).__init__()
        self.focal_gamma = focal_gamma
        self.regr_weight = regr_weight
        self.lmks_weight = lmks_weight

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[1], req[1], in_data[1])
        self.assign(out_data[2], req[2], in_data[2])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        prob = in_data[0]
        regr = in_data[1]
        lmks = in_data[2]

        prob_label = in_data[3]
        regr_label = in_data[4]
        lmks_label = in_data[5]

        regr_mask = prob_label != 0
        lmks_mask = prob_label == -2

        # focal loss
        gamma = float(self.focal_gamma)
        prob = mx.nd.clip(prob, 1e-6, 1.0 - 1e-6)
        grad = ((1.0 - prob)**gamma)*(-1.0/prob + gamma*mx.nd.log(prob)/(1-prob))
        self.assign(in_grad[0], req[0], grad*mx.nd.one_hot(prob_label, 2))

        # soft l1 loss: regr
        regr_weight = float(self.regr_weight)
        regr_delta = regr - regr_label
        regr_delta_abs = mx.nd.abs(regr_delta)
        regr_grad = regr_delta * (regr_delta_abs < 1.0) + (regr_delta_abs > 1.0)*((regr_delta > 0)-(regr_delta < 0))
        self.assign(in_grad[1], req[1], regr_weight * regr_grad * regr_mask.reshape((-1, 1)))

        # soft l1 loss: lmks
        lmks_weight = float(self.lmks_weight)
        lmks_delta = lmks - lmks_label
        lmks_delta_abs = mx.nd.abs(lmks_delta)
        lmks_grad = lmks_delta * (lmks_delta_abs < 1.0) + (lmks_delta_abs > 1.0) * ((lmks_delta > 0) - (lmks_delta < 0))
        self.assign(in_grad[2], req[2], lmks_weight * lmks_grad * lmks_mask.reshape((-1, 1)))


@mx.operator.register('MtcnnOutput')
class MtcnnOutputProp(mx.operator.CustomOpProp):
    def __init__(self, focal_gamma, regr_weight, lmks_weight):
        super(MtcnnOutputProp, self).__init__(need_top_grad=False)
        self.focal_gamma = focal_gamma
        self.regr_weight = regr_weight
        self.lmks_weight = lmks_weight

    def list_arguments(self):
        return ['prob', 'regr', 'lmks', 'prob_label', 'regr_label', 'lmks_label']

    def list_outputs(self):
        return ['prob', 'regr', 'lmks']

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]

        assert in_shape[0][1] == 2
        assert in_shape[1][0] == batch_size
        assert in_shape[1][1] == 4
        assert in_shape[2][1] == lmk_cnt * 2

        return [in_shape[0], in_shape[1], in_shape[2], (batch_size,), (batch_size, 4), (batch_size, lmk_cnt * 2)], \
               [in_shape[0], in_shape[1], in_shape[2]], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(self.list_arguments()), [dtype] * len(self.list_outputs())

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return MtcnnOutput(self.focal_gamma, self.regr_weight, self.lmks_weight)
