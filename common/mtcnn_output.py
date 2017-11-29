
import mxnet as mx


class MtcnnOutput(mx.operator.CustomOp):
    def __init__(self):
        super(MtcnnOutput, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[1], req[1], in_data[1])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        prob = in_data[0]
        regr = in_data[1]

        prob_label = in_data[2]
        regr_label = in_data[3]

        regr_mask = prob_label != 0

        pos_mask = prob_label == 1
        neg_mask = prob_label == 0

        loss = mx.nd.sum(-mx.nd.log(prob)*mx.nd.one_hot(prob_label, 2), axis=1)
        sum_pos = mx.nd.sum(pos_mask)
        sum_neg = mx.nd.sum(neg_mask)
        sum_pos_neg = (sum_pos + sum_neg).asnumpy()
        hard_example_mask = mx.nd.topk(loss, k=int(0.7*sum_pos_neg), ret_typ='mask')

        # prob_mask = pos_mask * (sum_pos_neg / sum_pos * 0.5) + neg_mask * (sum_pos_neg / sum_neg * 0.5)
        # prob_mask = pos_mask * 2.0 * (1 - prob[:, 1]) + neg_mask * (1 - prob[:, 0])
        # prob_mask = pos_mask * prob_mask * (prob[:, 1] < 0.9) + neg_mask * prob_mask * (prob[:, 0] < 0.9)
        # prob_mask = pos_mask*1.5 + neg_mask*0.7
        # prob_mask = pos_mask * 2.7 + neg_mask * 0.67
        # prob_mask = pos_mask * 1.5 + neg_mask * 0.6
        # prob_mask = pos_mask * 1.0 + neg_mask * 1.0

        # self.assign(in_grad[0], req[0], -1.0/prob*mx.nd.one_hot(prob_label, 2)*prob_mask.reshape((-1, 1)))
        self.assign(in_grad[0], req[0], -1.0/prob*mx.nd.one_hot(prob_label, 2)*hard_example_mask.reshape((-1, 1)))
        self.assign(in_grad[1], req[1], 0.5 * (regr - regr_label) * regr_mask.reshape((-1, 1)))


@mx.operator.register('MtcnnOutput')
class MtcnnOutputProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MtcnnOutputProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['prob', 'regr', 'prob_label', 'regr_label']

    def list_outputs(self):
        return ['prob', 'regr']

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]

        assert in_shape[0][1] == 2
        assert in_shape[1][0] == batch_size
        assert in_shape[1][1] == 4

        return [in_shape[0], in_shape[1], (batch_size, ), (batch_size, 4)], [in_shape[0], in_shape[1]], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(self.list_arguments()), [dtype] * len(self.list_outputs())

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return MtcnnOutput()

