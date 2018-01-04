import hyper_params.wider_face as ds

gpus = (0, 1)

# symbol config
focal_gamma = 0.5
regr_weight = 1.5

# learning config
learning_rate = 0.1
num_epoch = 130
epoch_steps = (44, 52, 56)

# dataset config
input_size = 24
batch_size = 1024

iou_thresh = 0.65
neg_per_face = 50
pos_per_face = 20

filters = (
    # (
    #     '/home/yetiancai/workspace/mtcnn-mxnet/model/pnet/pnet-symbol.json',
    #     '/home/yetiancai/workspace/mtcnn-mxnet/model/pnet/pnet-0100.params',
    #     12, 0.1,
    # ),
)

img_root = ds.img_root
bbox_train = ds.bbox_train
bbox_val = ds.bbox_val
img_cnt = ds.estimate_img_cnt(neg_per_face, pos_per_face)
