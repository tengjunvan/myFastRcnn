# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:52:14 2020

@author: tengjunwan
"""
import numpy as np

class ProposalTargetCreator():
    """
    给ROIs给予相应的gt_boxes,以便在之后的fast_rnn网络中进行计算，注意target意味着
    在接下来的计算中作为training中的“真实”而通过loss对模型进行“指导”，即在模型的
    test/predict过程中这里是不需要的。
    参数:
        n_sample(int):从ProposalCreator输出的的roi中选取n_sample个数，进行label
        pos_ratio(float):正样本(非背景)的比例
        pos_iou_thresh(float):成为“正样本”的iou最低值
        neg_iou_thresh_hi(float):成为“背景”的范围值[lo, hi]的hi
        neg_iou_thresh_lo(float):成为“背景”的范围值[lo, hi]的lo
    """
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0,
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        
    def __call__(self,
                 rois,
                 gt_boxes,
                 labels,
                 #loc_normalize_mean=(0.0, 0.0, 0.0, 0.0),
                 #loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
                 ):
        """
        参数:
            rois(array):维度(#roi, 4),其中#roi<=pos_nms_nums,在训练阶段是2000
            gt_boxes(array):维度(#gt_boxes, 4),是输入图片的数据值
            labels(array):维度(#gt_boxes,),图片数据值，如[1,4]
            loc_normalize_mean(tuple of float):?他源代码似乎涉及到放大缩小，不管
            loc_normalize_std(tuple of float):?他源代码似乎涉及到放大缩小，不管
        返回:
            sample_rois(array):输入的roi的子集(从其中取样的roi),维度(n_sample, 4)
            gt_roi_locs(array):sample_roi和其对应的gt_box的loc值，维度同上，如果是
                背景的话，怎么搞？
            gt_roi_labels(array):sample_roi对应的label(n_sample,),其中label=0表示
                背景，1表示第一个class="person"，等等。
        """
        pos_num = np.round(self.n_sample * self.pos_ratio)  # 128*0.25=32
        
        # 计算roi和gt_boxes的ious
        ious = cal_ious(rois, gt_boxes)  # array(#rois,#gt_boxes)
        gt_assignment = ious.argmax(axis=1)  #一维array表示每行最大值所在的列
        max_ious = ious.max(axis=1)  # 返回每一行最大值
        
        # 首先不考虑iou满不满足条件，都标记对应的gt_boxes的labels
        gt_roi_labels = labels[gt_assignment]  # 一维array(#rois,)表示labels
        
        # 筛选出其中iou满足阈值的“foreground”
        # 最大为pos_num(当然会有不足的情况),所以真实的pos_num不一定是规定值
        pos_index = np.where(max_ious >=self.pos_iou_thresh)[0]
        pos_num_for_this_image = int(min(pos_num, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                    pos_index, size=pos_num_for_this_image, replace=False)
            
        # 筛选出其中iou不满足阈值的“background”，即[lo,hi]之间的值
        # 其个数为n_sample - (pos_num实际值=pos_num_for_this_image)
        # 当然如果你的lo,hi设置其他值会导致neg_num的实际值比上面的值小
        # 说明sample中的样本数量(正，负之和)有可能小于128
        neg_index = np.where((max_ious < self.neg_iou_thresh_hi) & 
                             (max_ious >= self.neg_iou_thresh_lo))[0]
        neg_num = self.n_sample - pos_num_for_this_image
        neg_num_for_this_image = int(min(neg_index.size, neg_num))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                    neg_index, size=neg_num_for_this_image, replace=False)
        # 最终标记“背景”为0    
        keep_index = np.append(pos_index, neg_index)
        gt_roi_labels = gt_roi_labels[keep_index]  # (n_sample,)
        gt_roi_labels[pos_num_for_this_image: ] = 0 # 背景标记为0
        sample_rois = rois[keep_index]  # (n_sample, 4)
        
        # 最后得到sample_rois和gt_boxes的loc值
        # gt_roi_locs 维度为(n_sample, 4)
        gt_roi_locs = get_locations_of_valid_anchors(sample_rois,
                                                     gt_assignment[keep_index],
                                                     gt_boxes)
        
        return sample_rois, gt_roi_labels, gt_roi_locs

#======helper function================
def cal_ious(anchors_inside, gt_boxes):
    """计算所有在界内的anchors和gt boxes的ious
    输入：anchors_inside,2d array 维度是(#index_inside,4)，所有在图片内的anchors
          gt_boxes: 2d array，维度是(#gt_boxes,4)
    输出：ious,2d array,维度(#index_inside, #gt_boxes),表示anchors_inside和相对
          应的gt_boxes的iou
    """
    if anchors_inside.shape[1] != 4 or gt_boxes.shape[1] != 4:
        raise IndexError
        
    # （xmin,ymin）中取最大的,重叠面积的左上角to_left
    # 这里是利用了broadcasting
    # tl的维度是（#index_inside,#gt_boxes,2）,2表示xmin,ymin
    tl = np.maximum(anchors_inside[:, None, :2], gt_boxes[:, :2])  
    # （xmax,ymax）中取最小的，重叠面积的右下角bottom_right
    # tl的维度是（#index_inside,#gt_boxes,2）,2表示xmax,ymax
    br = np.minimum(anchors_inside[:, None, 2:], gt_boxes[:, 2:])

    # 重叠面积area_i(#index_inside,#gt_boxes)
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)  
    # area_1,表示achors_inside的面积，1d array (#index_inside,)
    area_1 = np.prod(anchors_inside[:, 2:] - anchors_inside[:, :2], axis=1)
    # area_2,表示gt_boxes的面积，1d array (#gt_boxes,)
    area_2 = np.prod(gt_boxes[:, 2:] - gt_boxes[:, :2], axis=1)
    
#    print(area_1[:, None].shape)  (1072,1)
#    print((area_1[:, None] + area_2).shape)   (1072,2)
#    print((area_1[:, None] + area_2 - area_i).shape)  (1072,2)
    
    ious = area_i / (area_1[:, None] + area_2 - area_i)
  
    return ious

def get_locations_of_valid_anchors(valid_anchors, argmax_ious, gt_boxes):
    """给在界内的anchors（不考虑sample）进行对应的locations计算，locations的计算
    是每个anchor和其对应iou最大的gt_box之间的“差距值”：dx,dy,dw,dh
    输入:
        valid_anchors:anchors[index_inside]，2d array(#index_inside,4)，4表示
                      xmin,ymin,xmax,ymax
        argmax_ious:1d array(#index_inside,)表示每个anchor对应的iou最大的gt_box
                    的编号
        gt_boxes:2d array(#gt_boxes, 4),表示ground truth的xmin,ymin,xmax,ymax
    输出:
        anchor_locs:2d array(#index_inside, 4),第二维是dx,dy,dw,dh
    
    """
    max_iou_gt_boxes = gt_boxes[argmax_ious]
    
    width = valid_anchors[:, 2] - valid_anchors[:, 0]
    height = valid_anchors[:, 3] - valid_anchors[:, 1]
    ctr_x = valid_anchors[:, 0] + 0.5 * width 
    ctr_y = valid_anchors[:,1] + 0.5 * height
    
    
    base_width = max_iou_gt_boxes[:, 2] - max_iou_gt_boxes[:, 0]
    base_height = max_iou_gt_boxes[:, 3] - max_iou_gt_boxes[:, 1]
    base_ctr_x = max_iou_gt_boxes[:, 0] + 0.5 * base_width
    base_ctr_y = max_iou_gt_boxes[:, 1] + 0.5 * base_height
    
    eps = np.finfo(height.dtype).eps  # 最小值，防止除法溢出
    
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    
    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)
    
    # anchor_locs 2d array，(#index_inside,4)，4表示dx,dy,dw,dh
    anchor_locs = np.vstack((dx, dy, dw, dh)).transpose()
    
    return anchor_locs
        
if __name__ == "__main__": 
    
    from proposal_creator import ProposalCreator
    proposal_creator = ProposalCreator()
    
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
    rois = proposalcreator(rpn_locs, rpn_scores, anchors, size)

    print("rois shape:", rois.shape)  # (1653, 4)实际上不足2000，这种不足是在nms中导致的
    
    #=====上面是在proposal_creator中的测试代码==============
    proposal_target_creator = ProposalTargetCreator()
    
    gt_boxes = np.array([[82, 408, 331, 631],
                         [14,  20, 598, 846]], dtype="int32") # 第一张图gt_boxes
    
    labels = np.array([5, 1], dtype="int32")  # 第一张图的label编号
    
    sample_rois, gt_roi_labels, gt_roi_locs  = proposal_target_creator(rois,
                                                                       gt_boxes,
                                                                       labels)
    print("sample_rois shape:", sample_rois.shape)
    print("gt_roi_locs shape:", gt_roi_locs.shape)
    print("gt_roi_labels shape:", gt_roi_labels.shape)

       
        
            