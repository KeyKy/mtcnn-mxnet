
import mxnet as mx
import sys
import os
extra_path = os.path.realpath(os.path.join(__file__, '../../../gml/cmake-build-release/pygml/'))
print("adding extra path: ", extra_path)
sys.path.append(extra_path)
import pygml


class MtcnnRecordIter(mx.io.DataIter):
    def __init__(self, **kwargs):
        super(MtcnnRecordIter, self).__init__(batch_size=kwargs['batch_size'])
        self.batch_size = kwargs['batch_size']
        self.input_size = kwargs['input_size']

        self.source = pygml.FaceIter(**kwargs)
        self.reset()

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
        self.source.Reset()

    def next(self):
        if not self.source.Next():
            raise StopIteration()

        data = mx.nd.array(self.source.GetCurBatchData())
        label = self.source.GetCurBatchLabel()
        prob_label = mx.nd.array(label[:, 0])
        regr_label = mx.nd.array(label[:, 1:])

        return mx.io.DataBatch([data],
                               [prob_label, regr_label],
                               provide_data=self.provide_data,
                               provide_label=self.provide_label)
