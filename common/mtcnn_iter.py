
import mxnet as mx
import random


class MtcnnRecordIter(mx.io.DataIter):
    def __init__(self, batch_size, input_size, rand_mirror=False, **kwargs):
        super(MtcnnRecordIter, self).__init__(batch_size=batch_size)
        self.input_size = input_size
        self.rand_mirror = rand_mirror
        self.rec_iter = mx.io.ImageRecordIter(batch_size=batch_size, **kwargs)

    @property
    def provide_data(self):
        return [('data', (self.batch_size, 3, self.input_size, self.input_size)), ]

    @property
    def provide_label(self):
        return [
            ('prob_label', (self.batch_size, )),
            ('regr_label', (self.batch_size, 4))
        ]

    def reset(self):
        self.rec_iter.reset()

    def next(self):
        batch = next(self.rec_iter)

        data = batch.data[0]
        prob_label = batch.label[0][:, 0]
        regr_label = batch.label[0][:, 1:]

        if self.rand_mirror:
            mirror = random.random() > 0.5
        else:
            mirror = False

        if mirror:
            data = mx.nd.reverse(data, axis=3)
            x1 = regr_label[:, 0].copy()
            regr_label[:, 0] = -regr_label[:, 2]
            regr_label[:, 2] = -x1

        return mx.io.DataBatch([data],
                               [prob_label, regr_label],
                               provide_data=self.provide_data,
                               provide_label=self.provide_label)
