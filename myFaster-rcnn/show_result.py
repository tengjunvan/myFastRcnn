# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:33:16 2020

@author: tengjunwan
"""

from faster_rcnn_trainer import FasterRcnnTrainer
from model.faster_rcnn import FasterRcnn
from data.dataset2 import ImageDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
#from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter
from utils.draw_tool import draw_predict


"""临时文件，看看训练结果"""
# 载入数据
dataset = ImageDataset(csv_file='data/VOC_test_data.csv', 
                       image_root_dir='data/pic/test')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
# dataloader返回的数据为dict
#               {"img_tensor": img_tensor,   
#                "anchor_labels": anchor_labels,   
#                "anchor_locations": anchor_locations,  
#                "img_index": img_index,
#                "img_size": img_size,
#                "img_classes": img_classes,
#                "img_gt_boxes": img_gt_boxes} 
print("=========dataset loaded=========")

# 载入模型以及warper
path = 'model/pretrained_model/checkpoints/vgg16-397923af.pth' # vgg16地址
faster_rcnn = FasterRcnn(path)
trainer = FasterRcnnTrainer(faster_rcnn)
print("==========trainer loaded==========")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(torch.cuda.current_device())
#device = torch.device('cpu')
print("training device name is: ", device_name)
trainer = trainer.to(device)

# 如果已经训练了，载入之前的保存模型，并且设置已经跑到epoch数
# 注意如果保存时候已经model.to(device),这里先载入要在model.to(device)之后
already_trained = True

if already_trained==True:
    #already_trained_epoch = 4
    load_path = 'checkpoints/'+'fasterrcnn_02270332-epoch-10-trainloss-0.446-testloss-0.572'
    trainer.load(load_path)
    print("====saved model loaded====")
    
# 训练part
#loss_per_check_list = []
#total_epoch = 4 
#check_nums = 200  # 跑200个batch看下loss 

dir_path = 'data/pic/test/'
for i, sample in tqdm(enumerate(dataloader)):
    print(sample["img_index"])
    x = sample["img_tensor"].to(device)
    #gt_boxes = sample["img_gt_boxes"].to(device)
    #labels = sample["img_classes"].to(device)
    #gt_anchor_labels = sample["anchor_labels"].to(device)
    #gt_anchor_locs = sample["anchor_locations"].to(device)
    final_bboxes, labels, scores = trainer.faster_rcnn.predict(x)
    img_name = sample["img_index"][0]
    draw_predict(dir_path, img_name, final_bboxes, labels, scores)

        