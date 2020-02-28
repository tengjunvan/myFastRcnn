# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:57:44 2020

@author: tengjunwan
"""

import torch.nn as nn
import torch 
import numpy as np

class RoIPooling2D(nn.Module):
    """主要是通过内置的AdaptiveMaxPool2d完成roi pooling 工作
    输入:
        output_size(tuple of int):一般来说是(7,7)
        spatial_scale(float):一般来说1./16
        return_size(bool):一般来说是False
    """
    def __init__(self, output_size,
                 spatial_scale,
                 return_indices=False):
        super(RoIPooling2D, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.return_indices = return_indices
        self.adp_max_pool_2D = nn.AdaptiveMaxPool2d(output_size,
                                                      return_indices)
        
    def forward(self, x, rois):
        """
        原本的代码参考了陈云的代码，而他的代码也参考了别人的代码，显得很奇怪(需要
        roi_indices),而我这里一次只处理一张图片，所以是多余的操作，所以我又重写了一
        遍，主要是:
            1.减去roi_indices这样的操作
            2.尝试着去除for循环(貌似做不到)
        参数:
            x(tensor):维度(1,c,w,h)，一般来说在vgg16中是1,c=512,W/16,H/16;
            rois(numpy):维度(#sample_rois, 4),是由proposalcreator传来的经过nms的
                rois,在训练阶段这里#rois = 2000;
        """
        rois_ = torch.from_numpy(rois).float()  # 转化为tensor
          
        rois = rois_.mul(self.spatial_scale) # Subsampling ratio
        rois = rois.long()  # 这一步只会保留整数部分，并不是四舍五入
        #print(rois)
        num_rois = rois.size(0)  # 总共有多少个roi，这里是2000
        output = []
        for i in range(num_rois):
            roi = rois[i]  # roi维度(4，)
            im = x[..., roi[0]:(roi[2]+1), roi[1]:(roi[3]+1)]  # (1,c,roi_w,roi_h)
            #print(im[0,0,:,:])
            try:
                output.append(self.adp_max_pool_2D(im))  # 元素维度 (1,c,7,7)
            except RuntimeError:
                print("roi:", roi)
                print("raw roi:", rois_[i])
                print("im:", im)
                print("outcome:",self.adp_max_pool_2D(im))
        
        output = torch.cat(output, 0)  # output 维度(128, c=512, 7,7)
        return output
    
if __name__ == "__main__":
    x = torch.randn(1, 3, 50, 50)
    sample_rois = np.array([[0, 0, 17, 17],
                            [0, 0, 31, 31,],
                            [0, 0, 64, 64],
                            [0, 0, 128, 128]], dtype="float32")
    roi_pooling_layer = RoIPooling2D((7, 7), 1./16)
    output = roi_pooling_layer(x, sample_rois)