import os
import shutil
import subprocess


idl_root = '/home/yetiancai/data/idl'
dst_rec_dir = '/home/yetiancai/rec/idl'

# wider_face_root = '/home/yetiancai/data/wider-face'
# dst_rec_dir = '/home/yetiancai/rec/wider-face'

val_save_dir = os.path.join(idl_root, 'val_crops')
train_save_dir = os.path.join(idl_root, 'train_crops')


subprocess.check_call(["python",
                       '/home/yetiancai/repo/mxnet_tcye/tools/im2rec.py',
                       "--shuffle", "True", "--pack-label", "True", "--num-thread", "10",
                       os.path.join(val_save_dir, 'train'), val_save_dir,
])

subprocess.check_call(["python",
                       '/home/yetiancai/repo/mxnet_tcye/tools/im2rec.py',
                       "--shuffle", "True", "--pack-label", "True", "--num-thread", "10",
                       os.path.join(train_save_dir, 'train'), train_save_dir,
                       ])

shutil.copy(os.path.join(val_save_dir, 'train.rec'), os.path.join(dst_rec_dir, 'val.rec'))
shutil.copy(os.path.join(val_save_dir, 'train.idx'), os.path.join(dst_rec_dir, 'val.idx'))

shutil.copy(os.path.join(train_save_dir, 'train.rec'), os.path.join(dst_rec_dir, 'train.rec'))
shutil.copy(os.path.join(train_save_dir, 'train.idx'), os.path.join(dst_rec_dir, 'train.idx'))
