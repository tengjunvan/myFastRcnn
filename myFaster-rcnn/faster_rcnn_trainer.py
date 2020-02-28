# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 00:10:21 2020

@author: tengjunwan
"""

import torch.nn as nn
import torch
import time
import os

class FasterRcnnTrainer(nn.Module):
    """为了方便训练的一个warper器,包裹了FasterRcnn网络,网络返回losses(namedTuple),
    同时添加了优化方法等，使得优化一步到位.
    losses包括:
        rpn_loc_loss:RPN网络的回归loss
        rpn_cls_loss:RPN网络的分类loss
        roi_loc_loss:Fast rcnn(RoI)网络的回归loss
        roi_cls_loss:Fast rcnn(RoI)网络的分类loss
        total_loss:上面四个loss之和
    
    参数:
        faster_rcnn(nn.Module):model.faster_rcnn.py中初始化的FasterRcnn网络
    """
    
    def __init__(self, faster_rcnn):
        super(FasterRcnnTrainer, self).__init__()
        # 网络结构
        self.faster_rcnn = faster_rcnn
        
        # 优化器
        self.optimizer = self.get_optimizer()
    
    def forward(self, x,  # 输入
                gt_boxes, labels,  # 需要计算fast rcnn的target的数据
                #gt_anchor_locs, gt_anchor_labels  # rpn的target
                ):
        """跑一边forward并返回losses
        
        参数:
            x(tensor):维度(1, 3, W, H)
            gt_boxes(tensor):维度(batch=1,#gt_boxes, 4),该图片中gt_boxes的坐标,
                坐标模式是(xmin,ymin,xmax,ymax)
            labels(tensor):维度(batch=1, #labels),该图片中gt_boxes对应的label
            gt_anchor_locs(tensor):维度(batch=1, #anchors, 4),每个anchor和其对应
                的gt_box的loc值
            gt_anchor_labels(tensor):维度(batch=1, #anchors),这里因为在我设置读取
                data的时候以及进行了sample,"+1"表示"正"样本,"0"表示"负"样本,"-1"
                表示"忽略"
        返回:
            losses(namedTurple):见此类下的说明
        """
        start = time.time()
        losses = self.faster_rcnn(x, 
                                  gt_boxes, 
                                  labels,
                                  #gt_anchor_locs, 
                                  #gt_anchor_labels
                                  )
        end = time.time()
        print("FasterRcnn forward 时间消耗: %s seconds"%(end-start))
        return losses
    
    def train_step(self, x, 
                   gt_boxes, labels,  # 需要计算fast rcnn的target的数据
                   #gt_anchor_locs, gt_anchor_labels  # rpn的target
                   ):
        """封装了forward()和optimize过程的方法,调用这个方法即等于
        0.optimizer.zero_grad()
        1.model.forwar()
        2.loss.backward()
        3.optimizer.step()
        
        参数:
            和forward的输入是一样的.
            
        返回:
            losses(namedTurple):同上
        """
        self.optimizer.zero_grad()
        #start = time.time()
        losses = self.forward(x, gt_boxes, labels,
                              #gt_anchor_locs, gt_anchor_labels
                              )
        #end = time.time()
        #print("FasterRcnn forward 时间消耗: %s seconds"%(end-start))
        start = time.time()
        losses.total_loss.backward()
        end = time.time()
        print("FasterRcnn backward 时间消耗: %s seconds"%(end-start))
        start = time.time()
        self.optimizer.step()
        end = time.time()
        print("FasterRcnn optimizer 时间消耗: %s seconds"%(end-start))
        
        return losses.total_loss.item()
    
    def get_optimizer(self, lr = 1e-3, weight_decay = 0.0005, lr_decay = 0.1,
                      use_adam=False):
        """调用此函数创建optimizer,主要封装了Adam和SGD两个optimizer,默认的optimizer
        是使用SGD.
        这里涉及到一些w和b不同的优化策略，所以用创建了params=[]，来适应Adam或者SGD
        的api这里named_parameters()是一个generator，会得到所有nested在FasterRcnn-
        Trainer内的nn.Module内的layer的权重值的名字(generator)和tensor,名字的命名
        格式类似于name="xxxxx.0.weight" 或者"xxxxx.0.bias"这里主要是weight和bias
        有不同的更新方案，所以才需要这样。
        
        参数:
            lr(float): learning rate.
            weight_decay(float):
            lr_decay(float): learning rate decay.
            use_adam(bool): 是,用adam，不是，用SGD
        """
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr*2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 
                                'lr': lr, 'weight_decay': weight_decay}]
        if use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer
    
    def scale_lr(self, decay=0.1):
        """调用次函数使得optimizer中的lr缩小0.1倍,一般来说源代码中的训练是在10次epoch
        之后使用"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer    
    
    def save(self, save_optimizer=True, save_path=None,
             epoch=None, avg_train_loss=None, avg_test_loss=None):
        """保存model,包括optimizer以及其他的信息，并返回存储的路径
        
        参数:
            save_optimizer(bool): 是否存optimizer
            save_path(string): 特殊要求的存储路径(默认在checkpoints文件夹内)
            epoch(int): 已经训练的epoch数
            avg_loss(float):和类下的说明中的losses一样
            
        返回:
            save_path(string): 保存的路径
        """
        save_dict = dict()
        
        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['epoch'] = epoch  
        save_dict['train_loss'] = avg_train_loss
        save_dict['test_loss'] = avg_test_loss
        
        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()
        
        if save_path is None:
            timestr = time.strftime('%m%d%H%M')  # 例如02201537
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            save_path += '-epoch-%d' % (epoch)  # 在文件名后加额外说明说明
            save_path += '-trainloss-%.3f' % (avg_train_loss)
            save_path += '-testloss-%.3f' % (avg_test_loss)
            
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        torch.save(save_dict, save_path)
        return save_path
        
    def load(self, path, load_optimizer=True):
        """读取model的state_dict,返回self
        """
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
            
            
    
if __name__ == "__main__":
    from model.faster_rcnn import FasterRcnn
    path = 'model/pretrained_model/checkpoints/vgg16-397923af.pth'
    faster_rcnn = FasterRcnn(path)
    trainer = FasterRcnnTrainer(faster_rcnn)
    
    if False:  # 测试get_optimizer()
        print(trainer.optimizer)
            
    if False:  # 测试 save() 和load()
        path = trainer.save(save_optimizer=True, 
                            save_path=None,
                            losses=(0.01,0.1,0.2,0.3,0.61),
                            epoch=100
                            )
        trainer = trainer.load(path, load_optimizer=True)
    
    if True:    # 看各部分的时间消耗
        from data.dataset2 import ImageDataset
        from torch.utils.data import DataLoader
        dataset = ImageDataset(csv_file='data/VOC_data_rescale_name2num.csv', 
                               image_root_dir='data/resize/JPEGImages')
        dataloader = DataLoader(dataset, batch_size=1, 
                                shuffle=True, num_workers=0)
        
        for i, sample in enumerate(dataloader):
            x = sample["img_tensor"]
            gt_boxes = sample["img_gt_boxes"]
            labels = sample["img_classes"]
            #gt_anchor_labels = sample["anchor_labels"]
            #gt_anchor_locs = sample["anchor_locations"]
        
            losses = trainer.train_step(x, 
                                        gt_boxes,
                                        labels,  
                                        #gt_anchor_locs, 
                                        #gt_anchor_labels
                                        )
            if i == 2:
                break
        