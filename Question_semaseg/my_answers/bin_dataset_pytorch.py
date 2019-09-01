import torch
import torch.nn.functional as F
import cv2
import argparse
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 64, 64
out_height, out_width = 64, 64
GPU = False
torch.manual_seed(0)

def data_load(path):
    xs = []
    ts = []
    paths = []

    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs.append(x)


            gt_path = path.replace('images', 'seg_images').replace('.jpg', '.png')
            gt = cv2.imread(gt_path)
            gt = cv2.resize(gt, (out_width, out_height), interpolation=cv2.INTER_NEAREST)

            t = np.zeros((out_width, out_width, 1), dtype=np.int)

            ind = (gt[..., 0] > 0)
            t[ind] = 1

            ts.append(t)
            paths.append(path)

            cv2.imshow('input', x)
            cv2.imshow('gt', gt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()




    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)

    xs = xs.transpose(0, 3, 2, 1)
    print(xs.shape)

    return xs, ts, paths

if __name__ == '__main__':
    xs, ts, paths = data_load('/home/naoki/Documents/DeepLearningMugenKnock/Dataset/train/images')
