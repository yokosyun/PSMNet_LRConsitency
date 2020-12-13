from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *
import sys
sys.path.append('../')
from utils.activations_autofn import MishAuto

Act = nn.ReLU
# Act = SwishAuto
# Act = MishAuto


class PSMNet(nn.Module):
    def __init__(self):
        super(PSMNet, self).__init__()
        self.feature_extraction = feature_extraction()

        self.maxpool = nn.MaxPool2d(2)


        in_channels = 64

        

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(inplace=True)
        )

        dilation = 1
        pad = 1

        self.bottom_1 = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3, stride=1, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels*8, in_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*4, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

   
 
        self.classify = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))


    def estimate_disparity(self, cost, height, width):

        print(cost.shape)
    
        down1 = self.down1(cost)

        print(down1.shape)
        
        down2 = self.maxpool(down1)
        print(down2.shape)
        down2 = self.down2(down2)
        print("down2.shape=",down2.shape)

        bottom_1 = self.maxpool(down2)
        print(bottom_1.shape)
        bottom_1 = self.bottom_1(bottom_1)
        print("bottom_1.shape=",bottom_1.shape)



        up2 = F.interpolate(bottom_1, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        up2 = torch.cat([up2, down2], axis=1)
        up2 = self.up2(up2)

    
        up1 = F.interpolate(up2, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        up1 = torch.cat([up1, down1], axis=1)
        up1 = self.up1(up1)

        lr_disp = self.classify(up1)

        lr_disp = F.upsample(lr_disp, [height,width],mode='bilinear')
        lr_disp = torch.sigmoid(lr_disp)
        lr_disp = lr_disp *width
        left_disp = lr_disp[:,0,:,:]
        right_disp = lr_disp[:,1,:,:]

        return left_disp,right_disp

    def forward(self, left, right):


        left_feature     = self.feature_extraction(left)
        right_feature  = self.feature_extraction(right)

        lr_feature = torch.cat([left_feature, right_feature], axis=1)
 
        pred_left,pred_right = self.estimate_disparity(lr_feature,left.size()[2],left.size()[3])

        return pred_left,pred_right