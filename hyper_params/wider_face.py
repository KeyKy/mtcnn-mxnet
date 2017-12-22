# wider face dataset config

img_root = '/home/yetiancai/data/wider-face/images/'
bbox_train = '/home/yetiancai/data/wider-face/bbox.train'
bbox_val = '/home/yetiancai/data/wider-face/bbox.val'

origin_image_cnt = 12880
face_per_img = 4.8


def estimate_img_cnt(neg_per_face, pos_per_face):
    return origin_image_cnt * (neg_per_face + pos_per_face) * face_per_img
