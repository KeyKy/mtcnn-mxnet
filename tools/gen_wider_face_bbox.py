

if 0:
    bbox_gt = 'E:/data_pkg/WiderFace/wider_face_split/wider_face_train_bbx_gt.txt'
    output = 'E:/wider-face/bbox.train'
else:
    bbox_gt = 'E:/data_pkg/WiderFace/wider_face_split/wider_face_val_bbx_gt.txt'
    output = 'E:/wider-face/bbox.val'


f = open(bbox_gt)
o = open(output, 'w')

sum_faces = 0

while 1:
    fname = f.readline().strip()
    if not fname:
        break
    cnt = int(f.readline().strip())

    resline = [fname, ]
    for i in range(cnt):
        info = f.readline().strip().split()
        x1 = float(info[0])
        y1 = float(info[1])
        x2 = x1 + float(info[2])
        y2 = y1 + float(info[3])
        resline.append('%.2f %.2f %.2f %.2f' % (x1, y1, x2, y2))
        sum_faces += 1
    o.write(' '.join(resline) + '\n')

print('sum faces: ', sum_faces)

f.close()
o.close()
