import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height, img_width = 236, 236 #572, 572
out_height, out_width = 52, 52 #388, 388
GPU = False
torch.manual_seed(0)


def crop_layer(layer, size):
    _, _, h, w = layer.size()
    _, _, _h, _w = size
    ph = int((h - _h) / 2)
    pw = int((w - _w) / 2)

    return layer[:, :, ph:ph+_h, pw:pw+_w]


class Mynet(torch.nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

        base = 64

        self.enc1 = torch.nn.Sequential()

        for i in range(2):
            f == 3 if i ==0 else base
            self.enc1.add_module
