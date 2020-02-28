# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:29:04 2020

@author: tengjunwan
"""
import numpy as np

def adjust_anchors(anchors, locs):
    """这个函数是将anchors通过rpn得到的locs进行校正，注意，这个校正是对所有的anchor
    进行的，不论输入的input的图片的anchor是否为正样本或者其他。
        anchors(array):维度为(#anchors, 4),4为(xmin,ymin,xmax,ymax)
        locs(array):维度为(#anchors, 4),4为rpn网络的输出(tx,ty,tw,th)
    """
    if anchors.shape[0] == 0:  # 如果传进来的anchors数量为0
        return np.zeros((0, 4), dtype=locs.dtype)  # 返回一个(0,4)维的array
    
    # 转换为（w,h,ctr_x,ctr_y）的形式
    anchors_width = anchors[:, 2] - anchors[:, 0]
    anchors_height = anchors[:, 3] - anchors[:, 1]
    anchors_ctr_x = anchors[:, 0] + 0.5 * anchors_width
    anchors_ctr_y = anchors[:, 1] + 0.5 * anchors_height
    
    tx = locs[:, 0]
    ty = locs[:, 1]
    tw = locs[:, 2]
    th = locs[:, 3]
    
    ctr_x = tx * anchors_width + anchors_ctr_x
    ctr_y = ty * anchors_height + anchors_ctr_y
    width = np.exp(tw) * anchors_width
    height =  np.exp(th) * anchors_height
    
    # 再将adjust后的boxes改为(xmin,ymin,xmax,ymax)的形式,取名为roi,region of 
    # interst
    roi = np.zeros(locs.shape, dtype=locs.dtype)  # 即(#anchors, 4)
    roi[:, 0] = ctr_x - 0.5 * width  # xmin
    roi[:, 2] = ctr_x + 0.5 * width  # xmax
    roi[:, 1] = ctr_y - 0.5 * height # ymin
    roi[:, 3] = ctr_y + 0.5 * height # ymax
    #print(roi.shape)
    return roi
    
    
def non_maximum_suppression(roi, thresh):
    """他原来的代码写的好鸡儿复杂，似乎涉及到了cuda代码(cupy)的编写，我这里的代码
    主要是网页上的代码的编写，不涉及到gpu计算
    参数:
        roi(array):维度(m,4),m表示在调节anchor之后进行loc后的剪切以及去除过小边的
            box的个数,m<=#anchors
        thresh(int):nms算法中去除"重叠一起"的roi的阈值
    """
    xmin = roi[:, 0]
    ymin = roi[:, 1]
    xmax = roi[:, 2]
    ymax = roi[:, 3]
    
    areas = (xmax - xmin) * (ymax - ymin)
    keep = []  # 初始化保留的roi的index的list
    order = np.arange(roi.shape[0])  # [0,1,2...,m-1]
    while order.size > 0:
        i = order[0]  # 取最大的score的roi的index,第一次遍历i=0
        keep.append(i)  # 收入keep中
        xx1 = np.maximum(xmin[i], xmin[order[1:]]) # (m-1,),m-1为剩下的roi
        xx2 = np.minimum(xmax[i], xmax[order[1:]])
        yy1 = np.maximum(ymin[i], ymin[order[1:]])
        yy2 = np.minimum(ymax[i], ymax[order[1:]])
        
        width = np.maximum(0.0, xx2 - xx1)
        height =  np.maximum(0.0, yy2 - yy1)
        inter = width * height  # (m-1,)挑出的roi和剩下的roi的重叠面积
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # (m-1,)
        
        idx = np.where(iou <= thresh)[0]  # 表示剩下的roi(不满足一定重叠率)
        order = order[1 + idx]  # 剔除score最大以及iou重叠率的roi剩下roi的index
    
    roi_after_nms = roi[keep]
    return roi_after_nms, keep
    
class ProposalCreator():
    """
    通过调用内置函数__call__来通过rpn网络的locs值来校正anchors的位置,主要是完成
    rpn网络之后的nms工作(non maximum supression)
    参数:
        nms_thresh:
        n_train_pre_nums(int):在train阶段，进行nms之前保留的top anchors的数量/。
        n_train_post_nms(int):在test阶段，进行nms之后保留的top boxes的数量(基于
            anchor的校正)。
        n_test_pre_nms(int):同理，test阶段不同的设置。
        n_test_post_nms(int):同理，test阶段不同的设置。
        min_size(int):如果修正后的boxes的最小边小于该值,舍弃,防止roipool层切割得到
           某个维度为0.
    """
    def __init__(self, 
                 parent_model,  # 传入rpn网络
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
    
    def __call__(self, locs, scores, anchors, img_size):
        """
        参数:
            locs(array):维度(#anchors, 4),从rpn网络的输出rpn_locs传入
            scores(array):维度(#anchors, ),从rpn网络的输出rpn_scores传入
            anchors(array):维度(#anchors, 4),从数据中读取
            img_size(tuple of ints):(宽，高),从数据中读取
        返回:
            roi(array):维度为(#roi, 4),4为(xmin,ymin,xmax,ymax),如果是在training
                阶段,#roi=2000,如果是eval阶段,#roi=300
        """
        # self.parent_model是rpn,如果rpn.eval()会使得rpn.training = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms  # 12000
            n_post_nms = self.n_train_post_nms  # 2000
        else:
            n_pre_nms = self.n_test_pre_nms  # 6000
            n_post_nms = self.n_test_post_nms # 300
       
        
        # 先默认训练模式(v.10该回了上面的判断)
#        n_pre_nms = self.n_train_pre_nms  # 12000
#        n_post_nms = self.n_train_post_nms  # 2000
        
        roi = adjust_anchors(anchors, locs)  # (#anchors, 4),(xmin,ymin,xmax,ymax)
        
        # 剪切,因为校正的关系，会出现出界的情况
        roi[:, [0, 2]] = np.clip(roi[:, [0, 2]], 0, img_size[0])  # x轴剪切
        roi[:, [1, 3]] = np.clip(roi[:, [1, 3]], 0, img_size[1])  # y轴剪切
        #print("check1", roi.shape)
        # 去除 高 或 宽 < min_size的roi
        min_size = self.min_size
        roi_width = roi[:, 2] - roi[:, 0]
        roi_height = roi[:, 3] - roi[:, 1]
        keep = np.where((roi_width >= min_size) &
                        (roi_height >= min_size))[0]  # 满足条件的行index
        roi = roi[keep, :]
        
        #print("check2", roi.shape)
        
        # 对roi通过rpn的cls scores进行排序
        scores = scores[:, 1]  # 取第二列为“是object”的评分
        scores = scores[keep]  # (#keep,) 这里 #keep <= #anchors
        order = scores.argsort()[::-1]  # 得到scores的下降排列的坐标
        order = order[: n_pre_nms]   # 保留固定数量 训练 12000
        #print(order.shape)
        roi = roi[order]  # 从scores高到低的排序的roi
        #print("check3", roi.shape)  # 这里还是12000
        # nms
        roi_after_nms, _ = non_maximum_suppression(roi, thresh=self.nms_thresh)
        roi_after_nms = roi_after_nms[:n_post_nms]  # 保留固定数量 训练2000
        return roi_after_nms  # roi
        
        

    
    
    
if __name__ == "__main__":
    path = 'pretrained_model/checkpoints/vgg16-397923af.pth'
    from vgg16_to_rpn import Vgg16ToRPN
    vgg16torpn = Vgg16ToRPN(path)
    import torch
    x = torch.rand((1, 3, 600, 850))  # 第一张图片的大小
        
    rpn_locs, rpn_scores, anchors = vgg16torpn(x)  # tensor,tensor,numpy
    # 除去第一个batch维度
    # 因为我这里propoal的大部分操作是numpy操作
    rpn_locs = rpn_locs[0].detach().numpy()
    rpn_scores = rpn_scores[0].detach().numpy()
    
    proposalcreator = ProposalCreator()
    size = (600, 850)
    roi = proposalcreator(rpn_locs, rpn_scores, anchors, size)

    print(roi)  # (1653, 4)实际上不足2000，这种不足是在nms中导致的