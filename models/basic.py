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
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

########
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     Act(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     Act(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Act(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Act(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Act(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Act(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Act(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

                
    def create_costvolume(self,refimg_fea, targetimg_fea):
        #matching (batch,32,maxdisparity//4,height//4,width//4)
        # cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp/4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_(), volatile= not self.training).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
                cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea

        cost = cost.contiguous()
        return cost

    def create_costvolume2(self,refimg_fea, targetimg_fea):
            #matching (batch,32,maxdisparity//4,height//4,width//4)
        # cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp/4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_(), volatile= not self.training).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,:refimg_fea.size()[3] - i]   = refimg_fea[:,:,:,:refimg_fea.size()[3] - i]
                cost[:, refimg_fea.size()[1]:, i, :,:refimg_fea.size()[3] - i] = targetimg_fea[:,:,:,i:refimg_fea.size()[3]]
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea

        cost = cost.contiguous()
        return cost


    def estimate_disparity(self, cost, height, width):
        # print("cost=",cost.shape)
        cost0 = self.dres0(cost)#this layer should hangle matching mainly
        cost0 = self.dres1(cost0) + cost0 #refinement
        cost0 = self.dres2(cost0) + cost0 #refinement
        cost0 = self.dres3(cost0) + cost0 #refinement
        cost0 = self.dres4(cost0) + cost0 #refinement

        cost = self.classify(cost0)#add feature to how the feature are high(weight should be all 1.0(3x3))
        cost = F.upsample(cost, [self.maxdisp,height,width],mode='trilinear') #upsample to original size
        # print("cost=",cost.shape)
        cost = torch.squeeze(cost,1)
        # print("cost=",cost.shape)
        #SoftArgMax
        pred = F.softmax(cost)
        pred = disparityregression(self.maxdisp)(pred)
        return pred

    def forward(self, left, right):

        left_feature     = self.feature_extraction(left)
        right_feature  = self.feature_extraction(right)
 
        #matching
        cost_left = self.create_costvolume(left_feature,right_feature)#batch,channel,disparity,height,width
        pred_left = self.estimate_disparity(cost_left,left.size()[2],left.size()[3])

        cost_right = self.create_costvolume2(right_feature,left_feature)
        pred_right = self.estimate_disparity(cost_right,left.size()[2],left.size()[3])

        return pred_left,pred_right