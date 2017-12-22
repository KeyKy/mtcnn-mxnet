import hyper_params.wider_face as ds

gpus = (3,)

# symbol config
focal_gamma = 0.5
regr_weight = 1.0

# learning config
learning_rate = 0.1
num_epoch = 20
epoch_steps = (10, 14, 18)

# dataset config
input_size = 48
batch_size = 1024

iou_thresh = 0.65
neg_per_face = 50
pos_per_face = 20

img_root = ds.img_root
bbox_train = ds.bbox_train
bbox_val = ds.bbox_val
img_cnt = ds.estimate_img_cnt(neg_per_face, pos_per_face)
