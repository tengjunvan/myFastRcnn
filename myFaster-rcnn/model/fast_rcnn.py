# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:29:07 2020

@author: tengjunwan
"""

#classes = ['person',
#       'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#       'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#       'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
#       ] 总共20，加上背景20+1=21

import torch.nn as nn
from model.roi_pooling_2d_v2 import RoIPooling2D
import time

"""
"""
class FastRcnn(nn.Module):
    """这里封装了fast rnn网络结构，主要是将rpn网络提供的roi“投射”到vgg16的feature
    map上，进行相应的切割并maxpooling(RoI maxpooling),再将其展开从2d变为1d,投入
    两个fc层(陈云代码里叫classifier),然后再分别带入两个分支fc层，作为cls和reg的输
    出
    参数:
        n_class(int):总共由多少个类，在我代码里面(voc2007数据)，总共20类，包括背景
            有21个类
        roi_size(tuple of int):roi pooling层之后的2d维度，原文中是7*7
        spatial_scale(float):roi(rpn推荐的区域，是原图上的区域)投射在feature map
           层后的需要缩小的比例,一般来说使用vgg16的值是1./16
        classifier(nn.Sequential):是从vgg16提取的两层fc层(relu激活)
    """
    
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(FastRcnn, self).__init__()
        
        # 这个是在fast_rcnn中的两层fcs(relu激活),roi pooling之后的维度为(512,7,7)
        # 展开为一维为(512*7*7,)=(25088,),作为第一个fc输入维度,输出维度为(4096,)
        # 第二个fc层输入和输出维度都为(4096,)
        # 一开始我是自己重新定义了classifier,发现没有写初始化，他原来代码里面这里
        # 是利用了vgg16已经训练好的两层fc,所以不需要写初始化
#        self.classifier = nn.Sequential(nn.Linear(in_features=25088,
#                                                  out_features=4096, bias=True),
#                                        nn.ReLU(inplace=True),
#                                        nn.Linear(in_features=4096,
#                                                  out_features=4096, bias=True),
#                                        nn.ReLU(inplace=True)
#                                        )
        self.classifier = classifier
        
        # 接下来两个fc分支分别作为fast_rcnn的“分类”和“回归”输出
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        
        # 这两个fc分支分别做如下初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        
        # 在输入fast_rcnn之前我们需要做roi pooling，下面就是定义了roi pooling层
        # 这里我暂时没看他的代码，只是逻辑上当作ROI max pooling layer
        # 后来我写了个基于adaptivemaxpooling2d的roi pooling层
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D((self.roi_size, self.roi_size),
                                self.spatial_scale)
        
    def forward(self, x, sample_rois):
        """
        参数:
            x(tensor):维度(1,c,w,h)
            rois(ndarray):维度(#rois, 4),来自proposal_target_creator
                的输出,这里#rois在是128
            # rois_indices:原来还有这个，是一次处理多个图片时候用的，但是他代码没
               有完成这个功能，是rpn网络的输出
        """
        #start = time.time()
        pool = self.roi(x, sample_rois)  #  tensor(#rois, c, 7, 7)
        #end = time.time()
        #print("fast rcnn: roi pool 时间消耗: %s seconds"%(end-start))
        #print(pool.shape)
        pool = pool.view(pool.size(0), -1)  # 展开2d变为1d
        #print(pool.shape)
        #start = time.time()
        fc7 = self.classifier(pool)  # 经过两个fc层(relu激活)
        #end = time.time()
        #print("fast rcnn: classifier 时间消耗: %s seconds"%(end-start))
        
        # 分别传入两个fc层作为classification和regression
        #start = time.time()
        roi_cls_locs = self.cls_loc(fc7)  
        roi_scores = self.score(fc7)
        #end = time.time()
        #print("fast rcnn: roi locs和scores 时间消耗: %s seconds"%(end-start))
        return roi_cls_locs, roi_scores
        
        
        
def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()        


if __name__ == "__main__":
    from vgg16_to_rpn import decomVgg16
    import numpy as np
    import torch
    path = 'pretrained_model/checkpoints/vgg16-397923af.pth'
    _, classifier = decomVgg16(path)
    fast_rcnn = FastRcnn(21, 7, 1./16, classifier)
    x = torch.randn(1, 512, 50, 50)
    sample_rois = np.array([[0, 0, 17, 17],
                            [0, 0, 31, 31,],
                            [0, 0, 64, 64],
                            [0, 0, 128, 128]], dtype="float32")
    roi_cls_locs, roi_scores = fast_rcnn(x, sample_rois)