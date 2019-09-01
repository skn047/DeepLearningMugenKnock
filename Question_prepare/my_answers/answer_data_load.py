import cv2
import numpy as np
from glob import glob

np.random.seed(0)

num_classees = 2
img_height, img_width = 64, 64

classes = ['akahara', 'madara']

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

            for i, cls in enumerate(classes):
                if cls in dir_path:
                    t = i

            ts.append(t)
            print(path)
            paths.append(path)

    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)

    xs = xs.transpose(0, 3, 2, 1)
    print(xs.shape)

    return xs, ts, paths

xs, ts, paths = data_load('/home/naoki/Documents/DeepLearningMugenKnock/Dataset/train/images')
