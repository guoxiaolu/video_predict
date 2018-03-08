import numpy as np
import csv
import random
import os

split_ratio = 0.9
label = ['id', 'playcount/delta_normalize', 'like/delta_normalize', 'total']
f = open('data/iqiyi_ad.csv', 'r')
fl = open('data/video_download_list.txt', 'r')

train_path = 'data/train.txt'
test_path = 'data/test.txt'
if os.path.exists(train_path):
    os.remove(train_path)
if os.path.exists(test_path):
    os.remove(test_path)
ftrain = open(train_path, 'a')
ftest = open(test_path, 'a')

lines = fl.readlines()
video_list = [line.strip() for line in lines]

gt_all = {}
gt = {}
gt_0 = []
all_v = []
reader = csv.reader(f)
for row in reader:
    if reader.line_num == 1:
        label_id = [row.index(l) for l in label]
    else:
        id = int(row[label_id[0]])
        total_v = float(row[label_id[3]])
        if '%05d.mp4'%id in video_list:
            gt_all[id] = [float(row[label_id[1]]), float(row[label_id[2]]), total_v]
            all_v.append(total_v)
            if total_v < 0.1:
                gt_0.append(id)

hist = np.histogram(all_v, range=(0,1))
num = np.sort(hist[0])[-2]
gt_0_sample = random.sample(gt_0, len(gt_0)-num)
for k,v in gt_all.items():
    if k in gt_0:
        continue
    else:
        gt[k] = v
keys = gt.keys()
random.shuffle(keys)
split_idx = int(split_ratio * len(keys))


for i, k in enumerate(keys):
    if i < split_idx:
        ftrain.write('%05d.mp4\t%f\t%f\t%f\n'%(k, gt[k][0], gt[k][1], gt[k][2]))
    else:
        ftest.write('%05d.mp4\t%f\t%f\t%f\n'%(k, gt[k][0], gt[k][1], gt[k][2]))

ftest.close()
ftrain.close()
fl.close()
f.close()