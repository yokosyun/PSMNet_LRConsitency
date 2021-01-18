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
from utils.loss import *
import time

Act = nn.ReLU
# Act = SwishAuto
# Act = MishAuto

# Norm = nn.BatchNorm2d
Norm = nn.InstanceNorm2d
# Norm = nn.GroupNorm
group = 16

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

        self.maxpool = nn.MaxPool2d(2)


        in_channels = 64
        

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            Norm(in_channels*2),
            # Norm(group,in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1),
            Norm(in_channels*2),
            # Norm(group,in_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, padding=1),
            Norm(in_channels*4),
            # Norm(group,in_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3, padding=1),
            Norm(in_channels*4),
            # Norm(group,in_channels*4),
            nn.ReLU(inplace=True)
        )

        dilation = 1
        pad = 1

        self.bottom_11 = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels, kernel_size=3, stride=1, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
            Norm(in_channels),
            # Norm(group,in_channels),
            nn.ReLU(inplace=True),
        )

        dilation = 3

        self.bottom_12 = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels, kernel_size=3, stride=1, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
            Norm(in_channels),
            # Norm(group,in_channels),
            nn.ReLU(inplace=True),
        )

        dilation = 5

        self.bottom_13 = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels, kernel_size=3, stride=1, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
            Norm(in_channels),
            # Norm(group,in_channels),
            nn.ReLU(inplace=True),
        )

        dilation = 7

        self.bottom_14 = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels, kernel_size=3, stride=1, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
            Norm(in_channels),
            # Norm(group,in_channels),
            nn.ReLU(inplace=True),
        )


        self.bottom_fuse = nn.Sequential(
            nn.Conv2d(in_channels*8, in_channels*4, kernel_size=3, padding=1),
            Norm(in_channels*4),
            # Norm(group,in_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*4, in_channels*4, kernel_size=1, padding=0),
            Norm(in_channels*4)
            # Norm(group,in_channels*4)
        )



        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels*8, in_channels*4, kernel_size=3, padding=1),
            Norm(in_channels*4),
            # Norm(group,in_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*4, in_channels*2, kernel_size=3, padding=1),
            Norm(in_channels*2),
            # Norm(group,in_channels*2),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels*2, kernel_size=3, padding=1),
            Norm(in_channels*2),
            # Norm(group,in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            Norm(in_channels),
            # Norm(group,in_channels),
            nn.ReLU(inplace=True)
        )

   
 
        # self.classify = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))


        self.classify = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, 1, kernel_size=3, padding=1))


        # self.refine = nn.Sequential(
        #     nn.Conv2d(in_channels/2+1, in_channels/2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels/2, 1, kernel_size=3, padding=1),
        # )

        self.refine = nn.Sequential(
            nn.Conv2d(7, 7, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, padding=1),
            nn.Conv2d(7, 1, kernel_size=3, padding=1),
        )





        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels/2, (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(in_channels/2),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels/2, in_channels/2, (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(in_channels/2),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels/2, in_channels/2, (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(in_channels/2),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv3d(in_channels/2, in_channels/2, (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(in_channels/2),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv3d(in_channels/2, in_channels/2, (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(in_channels/2),
            nn.ReLU(inplace=True)
        )
        self.branch6 = nn.Sequential(
            nn.Conv3d(in_channels/2, in_channels/2, (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(in_channels/2),
            nn.ReLU(inplace=True)
        )


        self.estDisp = nn.Sequential(
            nn.Conv2d(in_channels/2, in_channels/2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels/2, 1, kernel_size=3, padding=1),
        )








    # def estimate_disparity(self, cost, height, width):
    #     down1 = self.down1(cost)
    #     down2 = self.maxpool(down1)
    #     down2 = self.down2(down2)
    #     bottom_1 = self.maxpool(down2)
    
    #     bottom_11 = self.bottom_11(bottom_1)
    #     bottom_12 = self.bottom_12(bottom_1)
    #     bottom_13 = self.bottom_13(bottom_1)
    #     bottom_14 = self.bottom_14(bottom_1)

    #     bottom_out = torch.cat([bottom_1 ,bottom_11, bottom_12,bottom_13,bottom_14], axis=1)
    #     bottom_out = self.bottom_fuse(bottom_out)
       

    #     up2 = F.interpolate(bottom_out, size=None, scale_factor=2, mode='bilinear', align_corners=None)
    #     up2 = torch.cat([up2, down2], axis=1)
    #     up2 = self.up2(up2)
    
    #     up1 = F.interpolate(up2, size=None, scale_factor=2, mode='bilinear', align_corners=None)
    #     up1 = torch.cat([up1, down1], axis=1)
    #     up1 = self.up1(up1)

    #     return up1


    # def estimate_disparity(self, cost, height, width):

   
    #     down1 = self.down1(cost)
    #     down2 = self.maxpool(down1)
    #     down2 = self.down2(down2)
    #     bottom_1 = self.maxpool(down2)
    
    #     bottom_11 = self.bottom_11(bottom_1)
    #     bottom_12 = self.bottom_12(bottom_1)
    #     bottom_13 = self.bottom_13(bottom_1)
    #     bottom_14 = self.bottom_14(bottom_1)

    #     bottom_out = torch.cat([bottom_1 ,bottom_11, bottom_12,bottom_13,bottom_14], axis=1)
    #     bottom_out = self.bottom_fuse(bottom_out)
       

    #     up2 = F.interpolate(bottom_out, size=None, scale_factor=2, mode='bilinear', align_corners=None)
    #     up2 = torch.cat([up2, down2], axis=1)
    #     up2 = self.up2(up2)
    
    #     up1 = F.interpolate(up2, size=None, scale_factor=2, mode='bilinear', align_corners=None)
    #     up1 = torch.cat([up1, down1], axis=1)
    #     up1 = self.up1(up1)

    #     return up1


    def create_costvolume(self,refimg_fea, targetimg_fea):
        #matching (batch,32,maxdisparity//4,height//4,width//4)
        # cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()


        start_time = time.time()
        B, C, H, W = refimg_fea.shape
        cost = refimg_fea.new_zeros([B, 2*C, self.maxdisp//4, H, W])
        # cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp/4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_(), volatile= not self.training).cuda()
        # cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp/4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_(), volatile= not self.training).cuda()
        print('Variable = %.4f' %(time.time() - start_time))
        print("cost.shape=",cost.shape)
        print("refimg_fea.shape=",refimg_fea.shape)
        print("targetimg_fea.shape=",targetimg_fea.shape)


        start_time = time.time()
        for i in range(self.maxdisp//4):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
                cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
                

        print('copy = %.4f' %(time.time() - start_time))


        start_time = time.time()
        cost = cost.contiguous()
        print('contiguous time = %.4f' %(time.time() - start_time))

        return cost


    def estimate_disparity(self, cost, height, width):
        print("cost.shape=",cost.shape)
        branch1 = self.branch1(cost)
        print("branch1.shape=",branch1.shape)
        branch2 = self.branch2(branch1)
        print("branch2.shape=",branch2.shape)
        branch3 = self.branch3(branch2)
        print("branch3.shape=",branch3.shape)
        branch4 = self.branch4(branch3)
        print("branch4.shape=",branch4.shape)
        branch5 = self.branch5(branch4)
        print("branch5.shape=",branch5.shape)
        # branch6 = self.branch6(branch5)
        # print("branch6.shape=",branch6.shape)




        disp = self.estDisp(torch.squeeze(branch5,2))
        disp = torch.sigmoid(disp)* self.maxdisp
           
        return disp


    # def disparity_regression(self, input, height, width):
    
    #     lr_disp = self.classify(input)
    #     lr_disp = torch.sigmoid(lr_disp)
    #     lr_disp = lr_disp * self.maxdisp
    #     left_disp = lr_disp[:,0,:,:]
    #     right_disp = lr_disp[:,1,:,:]

    #     return left_disp,right_disp

    def disparity_regression(self, input, height, width):
        
        lr_disp = self.classify(input)
        lr_disp = torch.sigmoid(lr_disp)
        left_disp = lr_disp * self.maxdisp

        return left_disp



    # def refine_disparity(self, disprity,feature_map,height, width):         
    #     if disprity.ndim ==3:
    #         disprity = torch.unsqueeze(disprity,0)
    #     disprity = F.upsample(disprity, [height,width],mode='nearest')
    #     feature_map = F.upsample(feature_map, [height,width], mode='nearest')
    #     data = torch.cat([disprity, feature_map], axis=1)
    #     refined_disparity = self.refine(data)
    #     return refined_disparity


    def refine_disparity(self, disp_left,disp_right,left,right,height, width):         
        if disp_left.ndim ==3:
            disp_left = torch.unsqueeze(disp_left,0)
            disp_right = torch.unsqueeze(disp_right,0)
        disp_left = F.upsample(disp_left, [height,width],mode='nearest')
        disp_right = F.upsample(disp_right, [height,width],mode='nearest')

        loss_func = LRLoss()

        estRight = loss_func.bilinear_sampler_1d_h(left, disp_right)
        estLeft = loss_func.bilinear_sampler_1d_h(right, -1 * disp_left)
        
        # gray_left = loss_func.getGrayImage(left)
        # gray_right = loss_func.getGrayImage(right)
        # gray_estLeft = loss_func.getGrayImage(estLeft)
        # gray_esttRight = loss_func.getGrayImage(estRight)

        # # 1. IMAGE RECONNSTRUCTION loss
        # SAD_left = torch.mean(torch.abs(left - estLeft),1)
        # SAD_right = torch.mean(torch.abs(right - estRight),1)
        # if SAD_left.ndim ==3:
        #     SAD_left = torch.unsqueeze(SAD_left,0)
        #     SAD_right = torch.unsqueeze(SAD_right,0)


        # SSIM_left = loss_func.SSIM1(gray_left,gray_estLeft,3)
        # SSIM_right = loss_func.SSIM1(gray_right,gray_esttRight,3)

        # SSIM_left=torch.nn.functional.pad(SSIM_left, pad=(2,2,2,2), mode='constant', value=0)
        # SSIM_right=torch.nn.functional.pad(SSIM_right, pad=(2,2,2,2), mode='constant', value=0)

        # LtoR = self.bilinear_sampler_1d_h(disp_left, disp_right)
        # RtoL = self.bilinear_sampler_1d_h(disp_right, -1 * disp_left)
        # lr_left_loss = torch.mean(torch.abs(RtoL - disp_left))

        # print("disp_left.shape=",disp_left.shape)
        # print("SAD_left.shape=",SAD_left.shape)
        # print("SSIM_left.shape=",SSIM_left.shape)



        # errorLeft = torch.cat([disp_left, SAD_left, SSIM_left], axis=1)
        # errorRight = torch.cat([disp_right, SAD_left, SSIM_left], axis=1)

        errorLeft = torch.cat([disp_left, estLeft, left], axis=1)
        errorRight = torch.cat([disp_right, estRight, right], axis=1)

        refinedLeft = self.refine(errorLeft)
        refinedRight = self.refine(errorRight)


        return refinedLeft,refinedRight

    def forward(self, left, right):

        start_time = time.time()
        left_feature     = self.feature_extraction(left)
        right_feature  = self.feature_extraction(right)
        print('feature extraction time = %.4f' %(time.time() - start_time))


        # lr_feature = torch.cat([left_feature, right_feature], axis=1)
 
        # up1 = self.estimate_disparity(lr_feature,left.size()[2],left.size()[3])


        # pred_left,pred_right = self.disparity_regression(up1,left.size()[2],left.size()[3])
        # pred_left = self.disparity_regression(up1,left.size()[2],left.size()[3])




        start_time = time.time()

        cost_volume = self.create_costvolume(left_feature,right_feature)

        print('cost volume time = %.4f' %(time.time() - start_time))

        start_time = time.time()
        pred_left = self.estimate_disparity(cost_volume,left.size()[2],left.size()[3])

        print('estimate disparity time = %.4f' %(time.time() - start_time))



        # refined_left_disparity = self.refine_disparity(pred_left,left_feature,left.size()[2],left.size()[3])
        # refined_right_disparity = self.refine_disparity(pred_right,right_feature,left.size()[2],left.size()[3])


        # refined_left_disparity,refined_right_disparity = self.refine_disparity(pred_left,pred_right,left,right,left.size()[2],left.size()[3])
        # refined_left_disparity,refined_right_disparity = self.refine_disparity(pred_left,pred_right,left,right,left.size()[2],left.size()[3])


                


        refined_left_disparity = F.upsample(pred_left, [left.size()[2],left.size()[3]],mode='bilinear')


        
        # return pred_left,pred_right
        # return refined_left_disparity,refined_right_disparity,pred_left,pred_right
        return refined_left_disparity