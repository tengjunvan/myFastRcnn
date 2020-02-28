# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:09:38 2020

@author: tengjunwan
"""

import torch.nn as nn
from model.load_pretrainedVGG16 import load_vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.fast_rcnn import FastRcnn
from model.proposal_target_creator import ProposalTargetCreator
from model.anchor_target_creator import AnchorTargetCreator
import torch
import torch.nn.functional as F
from collections import namedtuple
from model.proposal_creator import adjust_anchors, non_maximum_suppression
import numpy as np
import time

def no_grad(f):
    """作为装饰器,改变函数使得函数在with torch.no_grad()下运行,这里是装饰predict
    方法
    """
    def new_f(*args,**kwargs):
        with torch.no_grad():
           return f(*args,**kwargs)
    return new_f

"""该模块我是重新写了vgg16_to_rpn.py,因为我第一遍只做了vgg16到rpn网络,这一次我做
完了整个一套faster rcnn
"""

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])    

class FasterRcnn(nn.Module):
    """暂时不做参数输入，都在里面自己写好了,注意这里的写FasterRcnn网络的是只是从
    输入到输出,不涉及到其他target数据的输入,在外面封装的时候再加入target以及loss
    """
    def __init__(self, path):
        super(FasterRcnn, self).__init__()
        # 分解vgg16,得到前端和两层fc(relu激活)
        self.extractor, classifier = decomVgg16(path)
        # rpn部分
        self.rpn = RegionProposalNetwork()
        # anchor_target_creator部分
        self.anchor_target_creator = AnchorTargetCreator()
        # roi采样部分
        self.sample_rois = ProposalTargetCreator()
        # fast_rcnn部分
        self.fast_rcnn = FastRcnn(n_class=21,
                                  roi_size=7,
                                  spatial_scale=1./16,
                                  classifier=classifier)
        
        # 系数,用来计算l1_smooth_loss
        self.rpn_sigma = 3.
        self.roi_sigma = 1.
        
    
    def forward(self, x,  # 输入
                gt_boxes, labels  # 需要计算fast rcnn的target的数据
                #gt_anchor_locs, gt_anchor_labels  # rpn的target
                ):
        """参数:
            x(tensor):维度(1, 3, W, H)
            gt_boxes(tensor):维度(batch=1,#gt_boxes, 4),该图片中gt_boxes的坐标,
                坐标模式是(xmin,ymin,xmax,ymax)
            labels(tensor):维度(batch=1, #labels),该图片中gt_boxes对应的label
            #gt_anchor_locs(tensor):维度(batch=1, #anchors, 4),每个anchor和其对应
                的gt_box的loc值
            #gt_anchor_labels(tensor):维度(batch=1, #anchors),这里因为在我设置读取
                data的时候以及进行了sample,"+1"表示"正"样本,"0"表示"负"样本,"-1"
                表示"忽略"
            返回:
                rpn_cls_loss(tensor):scalar, rpn网络分类loss
                rpn_loc_loss(tensor):scalar, rpn网络回归loss
                roi_cls_loss(tensor):scalar, fast rcnn网络分类loss
                roi_loc_loss(tensor):scalar, fast rcnn网络回归loss
                total_loss(tensor):scalar, 上面四个loss之和
                
        """
        # -----------------part 1: feature 提取部分----------------------
        # 输入x是(batchsize=1,channel=3,width=W,height=H)
        # start = time.time()
        h = self.extractor(x)  # h是(1,channel=512,ww,hh),其中ww/hh是feature维度
        #end = time.time()
        #print("extractor 时间消耗: %s seconds"%(end-start))
        
        # -----------------part 2: rpn部分(output_1)---------------------
        img_size = (x.size(2), x.size(3))  # 宽，高
        # rpn_locs(tensor): (n=1, #anchors, 4)
        # rpn_scores(tensor): (n=1, #anchros, 2)
        # anchors(numpy): (#anchors, 4)
        # rois(numpy): (#rois, 4),训练阶段#rois=2000
        #start = time.time()
        rpn_locs, rpn_scores, anchors, rois= self.rpn(h, img_size)
        #end = time.time()
        #print("rpn 时间消耗: %s seconds"%(end-start))
        
        # ----------------part 3: roi采样部分----------------------------
        # 不是所有的rpn推荐的roi都要进入fast rcnn，而是采样其中的128个
        # 同时得到这几个roi的target
        # 使用到proposal_target_creator.py中的采样
        # sample_rois(numpy): (128, 4)
        # gt_roi_locs(numpy): (128, 4),取样的roi对应的gt_boxes的loc值
        # gt_roi_labels(numpy): (128, ), 取样的roi是什么分类
        #start = time.time()
        sample_rois, gt_roi_labels, gt_roi_locs = self.sample_rois(rois,
                                                  gt_boxes[0].detach().cpu().numpy(),
                                                  labels[0].detach().cpu().numpy())
        #end = time.time()
        #print("roi采样 时间消耗: %s seconds"%(end-start))
        
        # ---------------part 4: fast rcnn(roi)部分(output_2)------------
        # sample_rois作为feature map的切割依据,将h进行切割输入fast rcnn部分
        # roi_cls_locs(tensor): (#sample_rois, n_class*4=84)
        # roi_scores(tensor): (#sample_rois, n_class=21)
        #start = time.time()
        roi_cls_locs, roi_scores = self.fast_rcnn(h, sample_rois)
        #end = time.time()
        #print("fast rcnn 时间消耗: %s seconds"%(end-start))
        
        #----------------RPN loss------------------
        # anchor target获得 维度分别为(#anchors，4)和(#anchors)
        gt_anchor_locs, gt_anchor_labels = self.anchor_target_creator(
                gt_boxes[0].detach().cpu().numpy(),
                anchors,
                img_size)
        gt_anchor_locs = torch.from_numpy(gt_anchor_locs).cuda()
        gt_anchor_labels = torch.from_numpy(gt_anchor_labels).long().cuda()
        # rpn分类loss,这函数里实际上的效果是把-1标记的去除之后再计算
        # rpn_cls_loss(tensor): (#anchors,)
        #print(rpn_scores[0])
        #print(gt_anchor_labels[0].long())
        rpn_cls_loss = F.cross_entropy(rpn_scores[0],       # tensor(#anchors,2)
                                       gt_anchor_labels, # tensor(#anchors,)
                                       ignore_index=-1)     # 忽略label=-1
        #print(rpn_cls_loss.shape)
        # rpn回归loss
        # rpn_loc_loss(tensor):scalar

        rpn_loc_loss = _loc_loss(rpn_locs[0],
                                 gt_anchor_locs,
                                 gt_anchor_labels,
                                 self.rpn_sigma)

        #----------------Fast rcnn(roi) loss----------
        # roi分类loss
        gt_roi_labels = torch.from_numpy(gt_roi_labels).long().cuda()  # 转tensor
        gt_roi_locs = torch.from_numpy(gt_roi_locs).float().cuda()     # 转tensor
        roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_labels)  # scalar
        # roi回归loss
        n_sample = roi_cls_locs.shape[0]
        roi_cls_locs = roi_cls_locs.view(n_sample, -1, 4)  #(#sample_rois,21,4)
        # slice对应的locs来进行loc loss计算
        roi_locs = roi_cls_locs[torch.arange(0, n_sample).long(),
                                gt_roi_labels]  #  (#sample_rois, 4)
        roi_loc_loss = _loc_loss(roi_locs.contiguous(),
                                 gt_roi_locs,
                                 gt_roi_labels,
                                 self.roi_sigma)
        
        # 打包好上面的loss,返回
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        
        return LossTuple(*losses)
    
    @no_grad
    def predict(self, x):
        """用来infer输入的,即得到图片中的bounding boxes,和对应的分类,以及分类的概
        率,注意的是infer的流程和train的forward流程是不完全一样的
        1.rpn的输出rois的数目更少:2000(train)变为300(infer)
        2.没有roi 采样部分，
        
        参数:
            x(tensor):(1,3,W,H),还是从dataloader里面出来的图片,所以我还是把图片的
                range设置为[0-1],并且进行了normalize:
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
        
        返回:
            bboxes(numpy):(#bboxes, 4)图片中预测的bounding boxes,这里坐标的模式是
                xmin,ymin,xmax,ymax。
            labels(numpy):(#bboxes,),labels的标记是区间是[0,L-1],L是非背景的总类别,
                在这里是20,即label中不包含背景
            scores(numpy):(#bboxes,),表示每个label的confidence
        """
        self.eval()  # 设定为eval状态，会影响到rpn部分的rois的输出个数
        #-----------------part 1: feature 提取部分----------------------
        h = self.extractor(x)
        img_size = (x.size(2), x.size(3))  # 宽，高
        # -----------------part 2: rpn部分--------------------------
        # rpn_locs(tensor): (n=1, #anchors, 4)
        # rpn_scores(tensor): (n=1, #anchros, 2)
        # anchors(numpy): (#anchors, 4)
        # rois(numpy): (#rois, 4),infer阶段#rois=300
        rpn_locs, rpn_scores, anchors, rois= self.rpn(h, img_size)
        # ---------------part 3: fast rcnn(roi)部分----------------
        # sample_rois作为feature map的切割依据,将h进行切割输入fast rcnn部分
        # roi_cls_locs(tensor): (#rois, n_class*4=84)
        # roi_scores(tensor): (#rois, n_class=21)
        roi_cls_locs, roi_scores = self.fast_rcnn(h, rois)
        #----------------part 4:bboxes生成部分--------------------
        n_sample = roi_cls_locs.shape[0]  # n_sample=#rois=300
        roi_cls_locs = roi_cls_locs.view(n_sample, -1, 4)  # tensor(#rois,21,4)
        rois = torch.from_numpy(rois).cuda()
        rois = rois.view(-1, 1, 4).expand_as(roi_cls_locs)  #tensor(#rois,21,4)
        # 依据locs值修正rois为bboxes, bboxes(numpy),维度(#roi*21=6300, 4)
        bboxes = adjust_anchors(rois.cpu().numpy().reshape((-1, 4)),
                                roi_cls_locs.cpu().numpy().reshape((-1, 4))
                                )
        bboxes = torch.from_numpy(bboxes).cuda()  # 变回tensor
        # 修剪bboxes中的坐标，使其落在图片内
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]]).clamp(min=0, max=img_size[0])
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]]).clamp(min=0, max=img_size[1])
        bboxes = bboxes.view(n_sample, -1)  #tensor(#roi, 21*4)
        # roi_scores通过softmax转换为概率
        prob = F.softmax(roi_scores, dim=1)  #tensor(#rois, n_class=21)
        #----------------part 5:筛选环节------------------------
        raw_bboxes = bboxes.cpu().numpy()
        raw_prob = prob.cpu().numpy()
        
        # final_bboxes(numpy): (#final_bboxes, 4)
        # labels(numpy): (#final_bboxes, )
        # scores(numpy): (#final_bboxes, )
        final_bboxes, labels, scores = self._suppress(raw_bboxes, raw_prob)
        
        self.train()  # infer阶段结束还是设为train
        return final_bboxes, labels, scores
        
    def _suppress(self, raw_bboxes, raw_prob):
        """
        对inference期间的结果进行筛选,首先满足一定的roi_score,然后经过nms,得到最终
        的结果
        
        参数:
            raw_bboxes(numpy):维度(#roi, 21*4)
            raw_prob(numpy):维度(#rois, n_class=21)
            
        返回:
            bbox(numpy): 维度(#bbox, 4),这里的#bbox是最后得到的bounding box的个数
            label(numpy): 维度(#bbox, ),label=0表示非背景的第一类物品
            score(numpy): 维度(#bbox, ),相应label的roi_score
            
        """
        # 两个参数在val和visualize时是不同的，我这里只想visualize
        score_thresh = 0.7
        nms_thresh = 0.3
        n_class = 21
        
        bbox = list()
        label = list()
        score = list()
        
        for i in range(1, 21):  # 这里我手动设置为21种类,0表示背景
            bbox_i = raw_bboxes.reshape((-1, n_class, 4)) #(#rois, 21, 4)
            bbox_i = bbox_i[:, i, :]  #(#rois, 1, 4)
            prob_i = raw_prob[:, i]  # (#rois,)
            mask = prob_i > score_thresh  # (#该类score值够高的rois,)
            bbox_i = bbox_i[mask]  # (#该类score值够高的rois, 4)
            prob_i = prob_i[mask]  # (#该类score值够高的rois,)
            order = prob_i.argsort()[::-1]  # (#该类score值够高的rois,)
            bbox_i = bbox_i[order]  # 按照score值从大到小排列
            
            # keep(numpy), (#keep, ), bboxe_i_after_nms(#keep, 4)
            bbox_i_after_nms, keep = non_maximum_suppression(bbox_i, 
                                                             nms_thresh)
            bbox.append(bbox_i_after_nms)
            #label_i: numpy(#keep,),其中数值全为i-1
            label_i = (i - 1) * np.ones((len(keep),))  
            label.append(label_i)
            score.append(prob_i[keep])  # numpy(#keep,)
        
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)  # (#bbox, 4)
        label = np.concatenate(label, axis=0).astype(np.int32)  # (#bbox,)
        score = np.concatenate(score, axis=0).astype(np.float32)# (#bbox,)
        return bbox, label, score
            
            
            
            
            
            
        

        
    
    
#-------------helper function------------------------    
def _loc_loss(pred_loc, gt_loc, gt_label, sigma):
    """可以用来计算fast rcnn(roi)部分的回归loss,但同时也可以用来计算rpn部分的回归
    损失,只对rpn中的“正”样本,以及roi中的非背景样本进行loc loss的计算
    参数:
        pred_loc(tensor):rpn时维度(#anchors, 4),roi时维度(#sample_rois=128, 4)
        gt_loc(tensor):rpn时维度(#anchors, 4),roi时维度(#sample_rois=128, 4)
        gt_label(tensor):rpn时维度(#anchors,),roi时维度(#sample_rois=128, )
        sigma(float):控制smooth_l1_loss的"拐点"参数,rpn时是1, roi时是3
    """
    in_weight = torch.zeros(gt_loc.shape).cuda()
    
    # 仅仅对正样本计算回归值(positive anchors或者positive rois)
    # (gt_label>0)返回一个true/false的一维tensor(#gt_label,)
    # .view(-1,1)返回一个二维tensor(#gt_label, 1)
    # .expand_as 返回一个二维tensor(#gt_label, 4)
    # 整个操作下来就是得到gt_loc相同维度的矩阵，然后把label=0的行标记为0，label>0
    # 的行标记为1,作为一个筛选矩阵而使用着
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    
    # 计算还未normalize的loc loss
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma) 
    # Normalized标准化, 对rpn来说是“正”“负”样本的个数,对roi来说是所有的样本个数
    loc_loss /= ((gt_label >=0).sum().float())
    return loc_loss

def _smooth_l1_loss(x, t, in_weight, sigma):
    """sigma是控制smooth_l1形状的参数,平方段在[-1/sigma, +1/simga]
       平方段:y = σ**2/2 * x**2, 其中x in [-1/σ, +1/σ]
       直线段:y = x - 0.5/σ, 其中x in [+1/σ, +∞] or [-∞, -1/σ]
       
       1.对rpn时区间具体为(-1,1) 
       2.对fast rcnn(roi)时区间具体为(-1/3, +1/3) 
       参数:
           x(tensor):(#x, 4)需要比对的两个量的其中之一
           t(tensor):(#x, 4)需要比对的两个量的其中之二
           in_weight(tensor):(#x ,4),筛选矩阵
           sigma(float):1/sigma表示x轴上的smooth_l1的"拐点"
    """
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)  # (#x, 4)
    abs_diff = diff.abs()  # (#x, 4)
    flag = (abs_diff.data < (1. / sigma2)).float()  # (#x, 4), 筛选矩阵
    y = (flag * (sigma2 / 2.) * (diff ** 2) +       # (#x, 4)
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()  # 矩阵中所有元素相加，（1，)

def decomVgg16(path):
    """切割并freeze vgg16，返回一个feature exactor和它的classifier(两层fc,relu激活)
    输入:
        path(string):已经训练好的vgg16网络路径
    """
    model = load_vgg16(path)
    #print(model)
    features = list(model.features)[:30]  # 提取前30层
    classifier = model.classifier  # 包含两岑fc，还有dropout层，relu层等等
    
    classifier = list(classifier)
    del classifier[6]
    use_drop_out = False  # 不用drop out层
    if not use_drop_out:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)
    
    # 前10层不的参数不动
    for layer in features[:10]:  
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier    
    
if __name__ == "__main__":
    # --------初始化网络------------
    path = 'pretrained_model/checkpoints/vgg16-397923af.pth'
    faster_rcnn = FasterRcnn(path)  
     # --------准备数据(第一张图片)------------
    """第一张图片的数据
    000001.jpg,"[600, 850, 3]","[5, 1]",
    "[[82, 408, 331, 631], [14, 20, 598, 846]]"
    """
    from data.dataset2 import ImageDataset
    from torch.utils.data import DataLoader
    csv_file = "../data/VOC_data_rescale_name2num.csv"
    image_root_dir = "../data/resize/JPEGImages"
    dataset = ImageDataset(csv_file, image_root_dir)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)
#    sample = next(iter(dataloader))
#    print(sample["img_index"])  # 000001.jpg
#    x = sample["img_tensor"]
#    gt_boxes = sample["img_gt_boxes"]
#    labels = sample["img_classes"]
#    gt_anchor_locs = sample["anchor_locations"]
#    gt_anchor_labels = sample["anchor_labels"]
#    
#    # ----------带入网络-------------
#    losses = faster_rcnn(x, gt_boxes, labels, gt_anchor_locs, gt_anchor_labels)
    
    for i, sample in enumerate(dataloader):
        print(sample["img_index"])  # 000001.jpg
        x = sample["img_tensor"]
        gt_boxes = sample["img_gt_boxes"]
        labels = sample["img_classes"]
        #gt_anchor_locs = sample["anchor_locations"]
        #gt_anchor_labels = sample["anchor_labels"]
        
        losses = faster_rcnn(x, gt_boxes, labels, 
                             #gt_anchor_locs, gt_anchor_labels
                             )
        if i==0:
            break

