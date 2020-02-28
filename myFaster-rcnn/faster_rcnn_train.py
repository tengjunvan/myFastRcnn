# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:41:15 2020

@author: tengjunwan
"""

from faster_rcnn_trainer import FasterRcnnTrainer
from model.faster_rcnn import FasterRcnn
from data.dataset2 import ImageDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def evaluate_test_data(test_dataloader, trainer):
    """这里我并没有使用VOC2007规定metric来评估,而只是从test dataset的loss来评估
    """
    test_loss = 0.0
    counter = 0
    for i, sample in tqdm(enumerate(test_dataloader)):
        print(sample["img_index"])
        x = sample["img_tensor"].to(device)
        gt_boxes = sample["img_gt_boxes"].to(device)
        labels = sample["img_classes"].to(device)
        loss = trainer(x, gt_boxes, labels).total_loss.item()
        test_loss += loss
        counter += 1
    avg_test_loss = test_loss / counter
    return avg_test_loss


# 载入数据
dataset = ImageDataset(csv_file='data/VOC_train_data.csv', 
                       image_root_dir='data/pic/train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
testset = ImageDataset(csv_file='data/VOC_test_data.csv', 
                       image_root_dir='data/pic/test')
test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
# dataloader返回的数据为dict
#               {"img_tensor": img_tensor,      图片的数值
#                "img_index": img_index,        图片的名字
#                "img_classes": img_classes,    图片中的类别编号
#                "img_gt_boxes": img_gt_boxes}  图片中的gt_boxes的坐标
print("=========dataset loaded=========")

# 载入模型, 以及整合了save和train,设置optimizor的trainer
path = 'model/pretrained_model/checkpoints/vgg16-397923af.pth' # vgg16地址
faster_rcnn = FasterRcnn(path)
trainer = FasterRcnnTrainer(faster_rcnn)
print("==========trainer loaded==========")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(torch.cuda.current_device())
#device = torch.device('cpu')  代码在GPU上训练,而不是cpu
print("training device name is: ", device_name)
trainer = trainer.to(device)

# 如果已经训练了，载入之前的保存模型，并且设置已经跑到epoch数
# 注意如果保存时候已经model.to(device),这里先载入要在model.to(device)之后
already_trained_epoch = 11

if already_trained_epoch != 0:
    file_name = "fasterrcnn_02271009-epoch-11-trainloss-0.403-testloss-0.576"
    load_path = 'checkpoints/'+file_name
    trainer.load(load_path)
    print("trained model loaded")
    print("loaded model lr: ", trainer.optimizer.param_groups[0]["lr"])  # 导入模型的学习率
# 调节学习率
change_lr = False
if change_lr:
    scale = 0.1  # new_lr = lr* scale
    trainer.scale_lr(scale)
    
print("current lr: ", trainer.optimizer.param_groups[0]["lr"])  # 当前学习率
# 训练part
total_epoch = 4   # 设置这次training epoch数量
check_nums = 200  # 跑200个batch看下平均loss 


for epoch in range(total_epoch):
    running_loss = 0.0  # 用来记录200runs的总loss
    epoch_loss = 0.0    # 用来记录一个epoch的总loss
    
    writer = SummaryWriter()  # tensorboard的记录器
    
    for i, sample in tqdm(enumerate(dataloader)):
        print(sample["img_index"])
        x = sample["img_tensor"].to(device)
        gt_boxes = sample["img_gt_boxes"].to(device)
        labels = sample["img_classes"].to(device)
        #gt_anchor_labels = sample["anchor_labels"].to(device)
        #gt_anchor_locs = sample["anchor_locations"].to(device)
        
        loss = trainer.train_step(x, 
                                  gt_boxes,
                                  labels,  
                                  )
        running_loss += loss
        epoch_loss += loss
        
        if (i + 1 ) % check_nums == 0:
            avg_running_loss = running_loss / check_nums
            print('[epoch: %d, run_times: %d]--loss: %.4f'%(
                    already_trained_epoch+epoch+1, i+1, avg_running_loss))
            writer.add_scalar("train loss of 1 epoch", 
                              avg_running_loss,
                              i+1)
            #loss_per_check_list.append(avg_running_loss)
            running_loss = 0.0
            
    avg_epoch_loss = epoch_loss / len(dataset)  # epoch平均loss
    writer.add_scalar("epoch loss", 
                      avg_epoch_loss, 
                      already_trained_epoch+epoch+1
                      )
    avg_test_loss = evaluate_test_data(test_dataloader, trainer)
    writer.add_scalar("test loss", 
                      avg_test_loss, 
                      already_trained_epoch+epoch+1
                      )
    print("=========one epoch complete======")
    print("average epoch loss is: %.4f"%(avg_epoch_loss))
    print("test loss is: %.4f"%(avg_test_loss))
    
    writer.close()
    trainer.save(save_optimizer=True,
                 epoch=already_trained_epoch+epoch+1, 
                 avg_train_loss=avg_epoch_loss,
                 avg_test_loss=avg_test_loss)
    print("============model saved===========")



