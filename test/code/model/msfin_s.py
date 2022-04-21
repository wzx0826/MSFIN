import torch.nn as nn
import torch
from model import common


def make_model(args, parent=False):
    return MSFIN_S(args)

# downscale=2
class DS2_DOWN(nn.Module):
    def __init__(self,n_feat, kernel_size, negval, conv1=common.default_conv_stride2):
        super(DS2_DOWN, self).__init__()

        down_body = []
        down_body.append(conv1(n_feat, n_feat, kernel_size))
        down_body.append(nn.LeakyReLU(negative_slope=negval, inplace=True))
        down_body.append(nn.Conv2d(n_feat, 2 * n_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.down = nn.Sequential(*down_body)

    def forward(self, x):
        y = self.down(x)
        return y

class DS2_UP(nn.Module):
    def __init__(self,n_feat):
        super(DS2_UP, self).__init__()

        up_body = []
        up_body.append(common.default_conv(2 * n_feat, 2 * 2 * n_feat, kernel_size=3, bias=True))
        up_body.append(nn.PixelShuffle(2))
        self.up = nn.Sequential(*up_body)

    def forward(self, x):
        y = self.up(x)
        return y

# downscale=4
class DS4_DOWN(nn.Module):
    def __init__(self,n_feat, kernel_size, negval, conv2=common.default_conv_stride2):
        super(DS4_DOWN, self).__init__()

        # downscale
        down_body1 = []
        down_body1.append(conv2(n_feat, n_feat, kernel_size))
        down_body1.append(nn.LeakyReLU(negative_slope=negval, inplace=True))
        down_body1.append(nn.Conv2d(n_feat, 2 * n_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.down1 = nn.Sequential(*down_body1)

        down_body2 = []
        down_body2.append(conv2(2 * n_feat, 4 * n_feat, kernel_size))
        down_body2.append(nn.LeakyReLU(negative_slope=negval, inplace=True))
        down_body2.append(nn.Conv2d(4 * n_feat, 4 * n_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.down2 = nn.Sequential(*down_body2)

    def forward(self, x):
        y1 = self.down1(x)  # 2
        y2 = self.down2(y1)  # 4
        return y2

class MSFIN_S(nn.Module):
    def __init__(self, args, conv=common.default_conv, reduction=4):
        super(MSFIN_S, self).__init__()

        self.opt = args
        self.scale = args.scale
        self.phase = len(args.scale)
        self.num_steps = args.num_steps
        #        n_resblocks = args.n_blocks
        n_feat = args.n_feats
        kernel_size = 3
        n_colors = 3
        negval = 0.2

        # 先进行一个上采样
        self.upsample = nn.Upsample(scale_factor=max(args.scale),
                                    mode='bilinear', align_corners=False)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        #DS0
        self.ds0_body = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                  res_scale=1)
        self.ds0_body2 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds0_body3 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds0_body4 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds0_body5 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                res_scale=1)

        #DS2
        self.ds2_rcab = common.RRCAB(conv, 2 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                  res_scale=1)
        self.ds2_rcab2 = common.RRCAB(conv, 2 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds2_rcab3 = common.RRCAB(conv, 2 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds2_rcab4 = common.RRCAB(conv, 2 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds2_rcab5 = common.RRCAB(conv, 2 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                res_scale=1)

        #DS4
        self.ds4_rcab = common.RRCAB(conv, 4 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                  res_scale=1)
        self.ds4_rcab1 = common.RRCAB(conv, 4 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds4_rcab1_2 = common.RRCAB(conv, 4 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                     res_scale=1)
        self.ds4_rcab1_3 = common.RRCAB(conv, 4 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                     res_scale=1)
        self.ds4_rcab1_4 = common.RRCAB(conv, 4 * n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                res_scale=1)

        up_body = []
        up_body.append(common.default_conv(4 * n_feat, 2 * 2 * n_feat, kernel_size=3, bias=True))
        up_body.append(nn.PixelShuffle(2))
        self.ds4_up4 = nn.Sequential(*up_body)

        self.ds4_rcab2 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds4_rcab3 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.ds4_rcab3_2 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                     res_scale=1)
        self.ds4_rcab3_3 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                     res_scale=1)
        self.ds4_rcab3_4 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                res_scale=1)
        up_body2 = []
        up_body2.append(common.default_conv(n_feat, 2 * 2 * n_feat, kernel_size=3, bias=True))
        up_body2.append(nn.PixelShuffle(2))
        # up_body2.append(common.default_conv(n_feat,3, kernel_size=3, bias=True))
        self.ds4_up5 = nn.Sequential(*up_body2)

        self.ds2_down = DS2_DOWN(n_feat, kernel_size,negval)
        self.ds2_up = DS2_UP(n_feat)

        self.ds4_down = DS4_DOWN(n_feat, kernel_size, negval)

        self.Fd21 = nn.ConvTranspose2d(4 * n_feat, 2 * n_feat, kernel_size=2, stride=2, padding=0,bias=True)    #Fd21,Fd22,up2
        self.Fd22 = nn.ConvTranspose2d(4 * n_feat, 2 * n_feat, kernel_size=2, stride=2, padding=0,bias=True)    #Fd21,Fd22,up2
        self.Fd23 = common.default_conv(n_feat, 2 * n_feat, 1)  # Fd23,Fd24
        self.Fd24 = common.default_conv(n_feat, 2 * n_feat, 1)  # Fd23,Fd24
        # self.Fd01_02 = nn.ConvTranspose2d(4 * n_feat, n_feat, kernel_size=4, stride=4, padding=0,bias=True)  # Fd01,Fd02,up4
        # self.Fd03_04 = nn.ConvTranspose2d(n_feat, n_feat, kernel_size=2, stride=2, padding=0,bias=True)  # Fd03,Fd04,up2
        self.Ft01 = nn.ConvTranspose2d(2 * n_feat, n_feat, kernel_size=2, stride=2, padding=0,bias=True)  # Ft01,Ft02Ft03,Ft04,up2
        self.Ft02 = nn.ConvTranspose2d(2 * n_feat, n_feat, kernel_size=2, stride=2, padding=0,bias=True)  # Ft01,Ft02Ft03,Ft04,up2
        self.Ft03 = nn.ConvTranspose2d(2 * n_feat, n_feat, kernel_size=2, stride=2, padding=0,bias=True)  # Ft01,Ft02Ft03,Ft04,up2
        self.Ft04 = nn.ConvTranspose2d(2 * n_feat, n_feat, kernel_size=2, stride=2, padding=0,bias=True)  # Ft01,Ft02Ft03,Ft04,up2

        self.block_re1 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                   res_scale=1)
        self.block_re2 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                    res_scale=1)
        self.block_re3 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                res_scale=1)
        self.block_re4 = common.RRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                                        res_scale=1)

        self.head = conv(n_colors, n_feat, kernel_size)
        self.tail2 = common.default_conv(n_feat, out_channels=3, kernel_size=3)
        self.c1 = common.default_conv(12*n_feat, out_channels=4*n_feat, kernel_size=1)
        self.c2 = common.default_conv(3 * n_feat, out_channels=n_feat, kernel_size=1)

    def forward(self, x):
        y1 = self.upsample(x)

        y_input1 = self.sub_mean(y1)
        y_input = self.head(y_input1)

        #DS4
        ds4_input = self.ds4_down(y_input)#4n
        DS4_1 = self.ds4_rcab(ds4_input)  # DS4_1
        for _ in range(self.num_steps):
            DS4_1 = self.ds4_rcab(DS4_1)
        DS4_2 = self.ds4_rcab1(DS4_1)     # DS4_2
        for _ in range(self.num_steps):
            DS4_2 = self.ds4_rcab1(DS4_2)#4n

        DS4_3 = self.ds4_rcab1_2(DS4_2)   # DS4_3
        for _ in range(self.num_steps):
            DS4_3 = self.ds4_rcab1_2(DS4_3)
        DS4_4 = self.ds4_rcab1_3(DS4_3)   # DS4_4
        for _ in range(self.num_steps):
            DS4_4 = self.ds4_rcab1_3(DS4_4)#4n

        DS4_concat1 = torch.cat([ds4_input,DS4_2,DS4_4],1)#12n
        mid1 = self.c1(DS4_concat1)

        DS4_4_1 = self.ds4_rcab1_4(mid1)
        for _ in range(self.num_steps):
            DS4_4_1 = self.ds4_rcab1_4(DS4_4_1)

        y_ds4_up1 = self.ds4_up4(DS4_4_1)

        DS4_5 = self.ds4_rcab2(y_ds4_up1) # DS4_5
        for _ in range(self.num_steps):
            DS4_5 = self.ds4_rcab2(DS4_5)
        DS4_6 = self.ds4_rcab3(DS4_5)     # DS4_6
        for _ in range(self.num_steps):
            DS4_6 = self.ds4_rcab3(DS4_6)

        DS4_7 = self.ds4_rcab3_2(DS4_6)   # DS4_7
        for _ in range(self.num_steps):
            DS4_7 = self.ds4_rcab3_2(DS4_7)
        DS4_8 = self.ds4_rcab3_3(DS4_7)   # DS4_8
        for _ in range(self.num_steps):
            DS4_8 = self.ds4_rcab3_3(DS4_8)

        DS4_concat2 = torch.cat([y_ds4_up1,DS4_6,DS4_8],1)#3n
        mid2 = self.c2(DS4_concat2)


        DS4_8_1 = self.ds4_rcab3_4(mid2)
        for _ in range(self.num_steps):
            DS4_8_1 = self.ds4_rcab3_4(DS4_8_1)

        ds4_out = self.ds4_up5(DS4_8_1)

        #DS2
        ds2_input = self.ds2_down(y_input)
        ds2_input1 = ds2_input + self.Fd21(DS4_2)

        DS2_1 = self.ds2_rcab(ds2_input1)   # DS2_1
        for _ in range(self.num_steps):
            DS2_1 = self.ds2_rcab(DS2_1)

        ds2_input2 = DS2_1 + self.Fd22(DS4_4)
        DS2_2 = self.ds2_rcab2(ds2_input2)  # DS2_2
        for _ in range(self.num_steps):
            DS2_2 = self.ds2_rcab2(DS2_2)

        ds2_input3 = DS2_2 + self.Fd23(DS4_6)
        DS2_3 = self.ds2_rcab3(ds2_input3)      # DS2_3
        for _ in range(self.num_steps):
            DS2_3 = self.ds2_rcab3(DS2_3)

        ds2_input4 = DS2_3 + self.Fd24(DS4_8)
        DS2_4 = self.ds2_rcab4(ds2_input4)         # DS2_4
        for _ in range(self.num_steps):
            DS2_4 = self.ds2_rcab4(DS2_4)

        DS2_5 = self.ds2_rcab5(DS2_4)
        for _ in range(self.num_steps):
            DS2_5 = self.ds2_rcab5(DS2_5)

        ds2_out = self.ds2_up(DS2_5)

        #DS0
        ds0_input = y_input
        # ds0_input1 = ds0_input + self.Fd01_02(DS4_2) + self.Ft01_04(DS2_1)
        ds0_input1 = ds0_input + self.Ft01(DS2_1)
        DS0_1 = self.ds0_body(ds0_input1)       # DS0_1
        for _ in range(self.num_steps):
            DS0_1 = self.ds0_body(DS0_1)

        # ds0_input2 = DS0_1 + self.Fd01_02(DS4_4) + self.Ft01_04(DS2_2)
        ds0_input2 = DS0_1  + self.Ft02(DS2_2)
        DS0_2 = self.ds0_body2(ds0_input2)      # DS0_2
        for _ in range(self.num_steps):
            DS0_2 = self.ds0_body2(DS0_2)

        # ds0_input3 = DS0_2 + self.Fd03_04(DS4_6) + self.Ft01_04(DS2_3)
        ds0_input3 = DS0_2 + self.Ft03(DS2_3)
        DS0_3 = self.ds0_body3(ds0_input3)      # DS0_3
        for _ in range(self.num_steps):
            DS0_3 = self.ds0_body3(DS0_3)

        # ds0_input4 = DS0_3 + self.Fd03_04(DS4_8) + self.Ft01_04(DS2_4)
        ds0_input4 = DS0_3  + self.Ft04(DS2_4)
        DS0_4 = self.ds0_body4(ds0_input4)      # DS0_4
        for _ in range(self.num_steps):
            DS0_4 = self.ds0_body4(DS0_4)

        DS0_5 = self.ds0_body5(DS0_4)
        for _ in range(self.num_steps):
            DS0_5 = self.ds0_body5(DS0_5)

        ds0_out = DS0_5
        #re
        out_a = ds0_out + ds2_out + ds4_out
        output1 = self.block_re1(out_a)
        for _ in range(self.num_steps):
            output1 = self.block_re1(output1)

        output1_2 = self.block_re2(output1)
        for _ in range(self.num_steps):
            output1_2 = self.block_re2(output1_2)

        output2 = self.block_re3(output1_2)
        for _ in range(self.num_steps):
            output2 = self.block_re3(output2)

        output3 = self.block_re4(output2)
        for _ in range(self.num_steps):
            output3 = self.block_re4(output3)

        output = self.tail2(output3)
        y = self.add_mean(output)

        return y