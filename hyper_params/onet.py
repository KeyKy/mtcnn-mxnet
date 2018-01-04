import hyper_params.wider_face as ds

gpus = (2, 3,)

# symbol config
focal_gamma = 0.5
regr_weight = 1.5
lmks_weight = 1.5

# learning config
learning_rate = 0.1
# num_epoch = 50
# epoch_steps = (20, 30, 40)
num_epoch = 60
epoch_steps = (20,)

# dataset config
input_size = 48
batch_size = 1024

iou_thresh = 0.65
neg_per_face = 50
pos_per_face = 20

filters = (
    (
        '/home/yetiancai/workspace/mtcnn-mxnet/model/pnet/pnet-symbol.json',
        '/home/yetiancai/workspace/mtcnn-mxnet/model/pnet/pnet-0100.params',
        12, 0.1,
    ),
    (
        '/home/yetiancai/workspace/mtcnn-mxnet/model/rnet/rnet-symbol.json',
        '/home/yetiancai/workspace/mtcnn-mxnet/model/rnet/rnet-0100.params',
        24, 0.05,
    ),
)

img_root = ds.img_root
bbox_train = ds.bbox_train
bbox_val = ds.bbox_val
img_cnt = ds.estimate_img_cnt(neg_per_face, pos_per_face)
