import numpy as np

sum_faces = 0

fn_lmk = 'E:/idl/annotations/all.test'
fn_bbox = 'E:/idl/annotations/bbox.val'

new_center_ind = 57

out_f = open(fn_bbox, 'w')
for line in open(fn_lmk, 'r').readlines():
    items = line.strip().split()
    lmks = np.array([float(s) for s in items[1::]]).reshape((-1, 72, 2))
    out_box_line = ''
    for i in range(lmks.shape[0]):
        box = None
        for j in range(lmks.shape[1]):
            if lmks[i, j, 0] == -1 and lmks[i, j, 1] == -1:
                continue
            if not box:
                box = [lmks[i, j, 0], lmks[i, j, 1], lmks[i, j, 0], lmks[i, j, 1]]
            else:
                box[0] = min(box[0], lmks[i, j, 0])
                box[1] = min(box[1], lmks[i, j, 1])
                box[2] = max(box[2], lmks[i, j, 0])
                box[3] = max(box[3], lmks[i, j, 1])
        if new_center_ind is not None:
            if lmks[i, new_center_ind, 0] == -1 and lmks[i, new_center_ind, 1] == -1:
                continue
            cx = lmks[i, new_center_ind, 0]
            cy = lmks[i, new_center_ind, 1]
        else:
            cx = (box[0] + box[2]) / 2.
            cy = (box[1] + box[3]) / 2.
        half_len = (box[3] - box[1] + box[2] - box[0]) / 4.
        box[0] = cx - half_len
        box[1] = cy - half_len
        box[2] = cx + half_len
        box[3] = cy + half_len
        out_box_line += ' '
        out_box_line += ' '.join(['%.2f' % b for b in box])

        sum_faces += 1

    out_f.write(items[0] + out_box_line + '\n')

out_f.close()
print("finished, sum faces: ", sum_faces)
