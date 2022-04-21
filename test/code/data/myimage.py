import os
import os.path
import random
import math
import errno

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0
        apath = args.testpath + '/' + args.testset + '/x' + str(args.scale[0])

        self.filelist = []
        self.imnamelist = []
        if not train:
            for f in os.listdir(apath):#列出apath路径下所有的文件夹或文件，LR图像
                try:
                    filename = os.path.join(apath, f)#图片路径
                    misc.imread(filename)#读取testpath/testset/xscale下的图片
                    self.filelist.append(filename)
                    self.imnamelist.append(f)
                except:
                    pass

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]#返回图片文件
        filename, _ = os.path.splitext(filename)#分离图像名与扩展名
        lr = misc.imread(self.filelist[idx])#读取lr图像
        lr = common.set_channel([lr], self.args.n_colors)[0]

        return common.np2Tensor([lr], self.args.rgb_range)[0], -1, filename
    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

