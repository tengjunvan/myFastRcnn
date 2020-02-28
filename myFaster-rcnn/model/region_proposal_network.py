# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:11:52 2020

@author: tengjunwan
"""
import numpy as np
from torch.nn import functional as F
import torch
from torch import nn
from model.proposal_creator import ProposalCreator

class RegionProposalNetwork(nn.Module): 
    """
    RPN网络
    参数:
        in_channels(int):
        mid_channels(int):
        ratios(list of floats):
        anchor_scales (list of numbers):
        feat_stride(int):
    """
    
    def __init__(self, 
                 in_channels=512, 
                 mid_channels=512,
                 feat_stride=16
                 ):
        super(RegionProposalNetwork, self).__init__()
        # 基本数值
        self.anchor_base = generate_anchor_base()  # 2d array(9,4)
        self.feat_stride = feat_stride  # vgg16为16
        n_anchor = self.anchor_base.shape[0]  # 一般为9
        
        # 可以把rpn传入，如果是train阶段，返回的roi数量是2000，如果是test则是300
        self.proposal_layer = ProposalCreator(parent_model=self)
        
        # layer层
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        
        # 伴随着类初始化而启动的函数(数初始化)，它封装了pytorch的初始化函数
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
        
    def forward(self, 
                x, 
                img_size,
                ):
        """
        输入:
            x(tensor):来自vgg16的part的输出,tensor(n=1,#channel,ww,hh)
            image_size(tuple of int):来自数据读取
        输出:
            rpn_locs(tensor):维度(n=1,#anchors,4)
            rpn_scores(tensor):维度(n=1,#anchors,2),第三维度的0代表“非”object的
                评分,1代表“是”object的评分
            anchors(array):维度(#anchors, 4)
            rois(array):train时维度(2000, 4), test时维度(300, 4)
        """
        n, _, ww, hh = x.shape
        anchors = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, ww, hh)  # array,(#anchors, 4)
    
        n_anchor = anchors.shape[0] // (hh * ww)  # 一般为9
        h = F.relu(self.conv1(x))
        
        # location回归part
        rpn_locs = self.loc(h)  # (n, 36, W, H)
        
        # 这里涉及contiguous概念，它报错的时候加上就好
        # rpn_locs.permute(0, 2, 3, 1)维度为(n,W,H,36)
        # ~.view(n, -1, 4) 维度为（n, anchors总数, 4）
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        
        # classifer分类part
        rpn_scores = self.score(h)  # (n, 18, W, H)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() #(n,W,H,18)
        rpn_scores = rpn_scores.view(n, ww, hh, n_anchor, 2) #(n,W,H,9,2)
        #rpn_softmax_scores = F.softmax(rpn_scores, dim=4) #(n,W,H,9,2)
        # 取第四维度的0为“否”，1为“是”，rpn_fg_scores表示anchor“是”的概率
        #rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  #(n,W,H,9)
        # (n, #anchors)
        #rpn_fg_scores = rpn_fg_scores.view(n, -1)  # (n, #anchors)
        rpn_scores = rpn_scores.view(n, -1, 2)  # (n, #anchors, 2) 还未softmax
        
        # 上面计算完了网络的输出值，但是还仍需要提供roi给fastrcnn部分
        # rois是np array,维度在train时是(2000, 4)
        rois = self.proposal_layer(rpn_locs[0].detach().cpu().numpy(),
                                   rpn_scores[0].detach().cpu().numpy(),
                                   anchors,
                                   img_size)
        
        return rpn_locs, rpn_scores, anchors, rois
    
def normal_init(m, mean, stddev, truncated=False):
    """
    weigth初始化
    """
    if truncated:  # 基本上用不到
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) 
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

def generate_anchor_base():
    """该函数生成中心坐标位于(8,8)所对应的9个anchor的坐标(xmin,ymin,xmax,ymax)
    返回:
        anchor_base:是一个2d array(9*4),其中9是分别对应9个anchor boxes，4是
        (xmin,ymin,xmax,ymax),
    """
    base_size =16.
    anchor_scales = [8, 16, 32]
    anchor_ratios = [0.5, 1, 2]
    
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((9, 4), dtype=np.float32)
    for i in range(len(anchor_scales)):  
        for j in range(len(anchor_ratios)):    
            """这样的算法分别对应面积为128**2，256**2，512**2"""
            """其中每个对应面积有0.5，1，2 的比例不同"""
            h = base_size * anchor_scales[i] * np.sqrt(anchor_ratios[j])
            w = base_size * anchor_scales[i] * np.sqrt(1. / anchor_ratios[j])

            index = i * 3 + j
            anchor_base[index, 0] = py - w / 2.
            anchor_base[index, 1] = px - h / 2.
            anchor_base[index, 2] = py + w / 2.
            anchor_base[index, 3] = px + h / 2.
            
    return anchor_base   

def _enumerate_shifted_anchor(anchor_base, feat_stride, width, height):
    """这个函数是generate_anchors的实际使用函数，主要是避免使用for循环,使用在
       region_proposal_network模块中
       参数:
           anchor_base(numpy 2d array,(9,4)):
           feat_stride(int):vgg16一般是16
           height:这里的height不是原图的height，而是feature map的height
           width:同理，这里也是feature map的height
       输出:
           anchors(numpy 2d array,(#anchor*9,4)):遍历顺序还是从每一行开始，从左
                 往右，然后从上往下进行遍历，与generate_anchors函数输出一样
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    # 下面利用了broadcasting机制，因为同一点的anchor的操作是一样的
    anchors = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32)
    return anchors      


if __name__ == "__main__":
    rpn = RegionProposalNetwork() 
    x = torch.rand((1, 512, 50, 50))
    img_size = (800, 800)
    rpn_locs, rpn_scores, anchors, rois = rpn(x, img_size)
    print("rpn_locs.shape:", rpn_locs.shape)  # tensor
    print("rpn_scores.shape", rpn_scores.shape)  # tensor 
    print("anchors.shape", anchors.shape)  # numpy
    print("rois.shape", rois.shape)