# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:02:41 2020

@author: tengjunwan
"""

import numpy as np

class AnchorTargetCreator():
    """之前在设置dataset的时候就提取了anchor_labels和anchor_locs,但因为从图片大小
    而推断feature层的大小的方法是有问题的，所以必须将feature层的大小作为输入来从新
    推断anchor_labels和anchor_locs,这里我引用原代码中的rpn中生成的anchors作为输入
    参数:
        n_sample(int):256,target的总数量
        pos_iou_thresh(float): 和gt_boxes的iou的阈值，超过此值为“正”样本,标记"1"
        neg_iou_thresh(float): 和gt_boxes的iou的阈值，低于此之为“负”样本,标记"0"
        pos_ratio(float): target总数量中"正"样本,如果正样本数量不足,则填充负样本
    """
    
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio
        
    def __call__(self, gt_boxes, anchors, img_size):
        """
        参数:
            gt_boxes(numpy):维度(#gt_boxes, 4),图片中真实框,(xmin,ymin,xmax,ymax)
            anchors(numpy):维度(#anchors, 4),rpn网络中输入
            img_size(tuple of ints):原图的(宽,高),用来筛选是否出界的anchors
        返回:
            anchor_locs(numpy):维度(#anchors, 4)
            anchor_labels(numpy):维度(#anchors,):正样本标记为"1",负样本标记为"0",
                忽略样本标记为"-1"
        """
        img_W, img_H = img_size
        
        n_anchor = len(anchors)
        inside_index = _get_inside_index(anchors, img_W, img_H)
        
        inside_anchors = anchors[inside_index]  #从原始的anchors得到图内的anchors
        
        #argmax_ious:(#index_inside,)每个inside_anchors对应gt_boxes的编号
        #labels:(#index_inside,)inside_anchors的label
        argmax_ious, labels = self._create_label(
                inside_index, inside_anchors, gt_boxes)
        
        # 计算inside_anchors和对应gt_box的回归值
        locs = bbox2loc(inside_anchors, gt_boxes[argmax_ious])
        
        # 把inside_anchors重新展开回原来所有的anchors
        anchor_labels = _unmap(labels, n_anchor, inside_index, fill=-1)
        anchor_locs = _unmap(locs, n_anchor, inside_index, fill=0)
        
        return anchor_locs, anchor_labels
        
    def _create_label(self, inside_index, inside_anchors, gt_boxes):
        """
        参数:
            inside_index:
            inside_anchors:
            gt_boxes:
        返回:
            argmax_ious(numpy):(#index_inside,),每个inside_anchors对应ious最大的
                gt_boxes的编号
            label(numpy):(#index_inside,),最终输出的+1和0为正负样本,其和一定是
                n_sample(即256)
        """
        # 先标记所有在图内的anchors为-1
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(inside_anchors, gt_boxes, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label
    
    def _calc_ious(self, inside_anchors, gt_boxes, inside_index):
        """
        输入:
            inside_anchors(numpy):(#index_inside,4) 在图内的anchor
            gt_boxes(numpy):(#gt_boxes, 4)
            side_index(numpy):(#index_inside,)标记原本anchors中的图内的行index
        返回:
            argmax_ious(numpy):(#index_inside,),每个inside_anchors对应ious最大的
                gt_boxes的编号
            max_ious(numpy):(#index_inside,),每个inside_anchor在所有gt_boxes中
                最大iou
            gt_argmax_ious(numpy):(#index_inside,),得到使得任意gt_box的iou最大
                inside_anchor的编号
        """
        ious = bbox_iou(inside_anchors, gt_boxes)  # (#index_inside, #gt_boxes)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious
    
# ------------帮助函数--------------        
def _get_inside_index(anchors, W, H):
    # 得到在图片内的anchors的index,anchors的坐标模式是(xmin,ymin,xmax,ymax)
    index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= W) &
        (anchors[:, 3] <= H)
    )[0]
    return index_inside  # numpy 维度(#anchors,)

def bbox_iou(anchors_inside, gt_boxes):
    """计算两个box的iou,涉及的numpy都是float32类型

    参数:
        anchors_inside(numpy):维度(N, 4),一般是anchors
        gt_boxes(numpy):维度(K, 4),一般是gt_boxes

    返回:
        ious(numpy):维度(N,K),这个表格记录了每一个box之间的iou
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

def bbox2loc(valid_anchors, max_iou_gt_boxes):
    """给在界内的anchors（不考虑sample）进行对应的locations计算，locations的计算
    是每个anchor和其对应iou最大的gt_box之间的“差距值”：dx,dy,dw,dh
    输入:
        valid_anchors:anchors[index_inside]，2d array(#index_inside,4)，4表示
                      xmin,ymin,xmax,ymax
        max_iou_gt_boxes:对应的gt_boxes
    输出:
        anchor_locs:2d array(#index_inside, 4),第二维是dx,dy,dw,dh
    
    """
    
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

def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

if __name__ == "__main__":
    
    from model.faster_rcnn import FasterRcnn
    from data.dataset import ImageDataset
    from torch.utils.data import DataLoader
    import torch
    
    path = 'pretrained_model/checkpoints/vgg16-397923af.pth'
    faster_rcnn = FasterRcnn(path)
    
    dataset = ImageDataset(csv_file='../data/VOC_data_rescale_name2num.csv', 
                           image_root_dir='../data/resize/JPEGImages')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    sample = next(iter(dataloader))
    x = sample["img_tensor"]
    gt_boxes = sample["img_gt_boxes"][0].detach().numpy()
    anchor_labels_1 = sample['anchor_labels'][0].detach().numpy()
    anchor_locs_1 = sample['anchor_locations'][0].detach().numpy()
    print(anchor_labels_1.shape)
    print(anchor_locs_1.shape)
    h = faster_rcnn.extractor(x)
    img_size = (x.size(2), x.size(3))
    _, _, anchors, _= faster_rcnn.rpn(h, img_size)
    print(anchors.shape)
    test = AnchorTargetCreator()
    anchor_locs_2, anchor_labels_2 = test(gt_boxes, anchors, img_size)
    print(anchor_labels_2.shape)
    print(anchor_locs_2.shape)

    