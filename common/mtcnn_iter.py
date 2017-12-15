
import mxnet as mx
import sys
import os
extra_path = os.path.realpath(os.path.join(__file__, '../../../gml/cmake-build-release/pygml/'))
print("adding extra path: ", extra_path)
sys.path.append(extra_path)
import pygml


class MtcnnRecordIter(mx.io.DataIter):
    def __init__(self, img_root, bbox_file, batch_size, input_size, neg_per_face, pos_per_face, img_per_batch, round_cnt=1):
        super(MtcnnRecordIter, self).__init__(batch_size=batch_size)
        self.batch_size = batch_size
        self.input_size = input_size
        self.round_cnt = round_cnt

        self.source = pygml.FaceIter(
            img_root,
            bbox_file,
            batch_size,
            input_size,
            neg_per_face,
            pos_per_face,
            img_per_batch,
        )

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
        self.cur_round_idx = 0
        self.source.Reset()

    def next(self):
        if not self.source.Next():
            self.cur_round_idx += 1
            if self.cur_round_idx >= self.round_cnt:
                raise StopIteration()
            else:
                self.source.Reset()
                return self.next()

        data = mx.nd.array(self.source.GetCurBatchData())
        label = self.source.GetCurBatchLabel()
        prob_label = mx.nd.array(label[:, 0])
        regr_label = mx.nd.array(label[:, 1:])

        return mx.io.DataBatch([data],
                               [prob_label, regr_label],
                               provide_data=self.provide_data,
                               provide_label=self.provide_label)
