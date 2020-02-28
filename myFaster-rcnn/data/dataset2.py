# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:16:26 2020

@author: tengjunwan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:59:56 2020

@author: tengjunwan
"""
import numpy as np
from PIL import Image
from ast import literal_eval
import pandas as pd
from torch.utils.data import Dataset
import torch 
#from data.sample_from_one_pic import get_anchor_labels_and_anchor_locs
from torchvision import transforms

# voc数据中二十类物品名，dataSet读取还是读相应的index,如person为1，bird为2
# 0表示背景
#classes = ['person',
#           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
#           ]

"""这是dataset的修改版，把anchor_locs和anchor_labels的取得改到了anchor_target_cr-
eator中，在这里取消，同时我还把size取小了，因为可以读x数据知道
"""

class ImageDataset(Dataset):
    def __init__(self,csv_file, image_root_dir, transform=None):
        """
        参数:
        """
        self.csv_file = pd.read_csv(csv_file)
        self.image_root_dir = image_root_dir
        self.total_nums_of_pics = len(self.csv_file)
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                    [0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
        
    def __len__(self):
        return self.total_nums_of_pics
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.csv_file.iloc[idx]
        img_index = data[0]
        #img_size = np.asarray(literal_eval(data[1])[:2])  # 宽*高

        img_classes = np.asarray(literal_eval(data[2]))  # 类列表
        img_gt_boxes = np.asarray(literal_eval(data[3]))  #list of [xmin,ymin,xmax,ymax]
      
        # 这里返回的两个值本来都是numpy的，但是经过了之后的dataloader之后，通通转
        # 为tensor，并都加上了batch size
        #anchor_labels, anchor_locations = get_anchor_labels_and_anchor_locs(data)
        path = self.image_root_dir + "/" + data[0]
        img = Image.open(path)
    
        img_tensor = self.transform(img)  # 默认是(C,H,W)的形式
        img_tensor = img_tensor.permute(0, 2, 1)  # 我用的(C,W,H)的形式
        return {"img_tensor": img_tensor,
                #"anchor_labels": anchor_labels,
                #"anchor_locations": anchor_locations, 
                "img_index": img_index,
                #"img_size": img_size,
                "img_classes": img_classes,
                "img_gt_boxes": img_gt_boxes} 

        


if __name__ == "__main__":
    csv_file = "VOC_data_rescale_name2num.csv"
    image_root_dir = "resize/JPEGImages"
    if False:
        dataset = ImageDataset(csv_file, image_root_dir)
    #    indices = torch.randperm(len(dataset)).tolist()
    #    dataset_train = Subset(dataset, indices[: -1000])
    #    dataset_test = Subset(dataset, indices[-1000:])
        print(dataset.total_nums_of_pics)  # 9963
        from torch.utils.data import DataLoader
        dataloader_train = DataLoader(dataset, batch_size=1, shuffle=False, 
                                      num_workers=0)
        for i_batch, sample in enumerate(dataloader_train):
            # sample是一个三个元素的tuple
            # 第一个元素是图片数据tensor，（1, 3, W, H）
            # 第二个元素是图片标记正负anchor样本的tensor,(1, #anchors)
            # 第三个元素是anchor的loc的tensor，其中非正样本的loc=0,(1, #anchors, 4)
            print(sample['img_tensor'].shape)
            print(sample['anchor_labels'].shape)
            print(sample['anchor_locations'].shape)
            print(sample['img_index'])
            print(sample['img_size'][0])
            print(sample['img_classes'][0])
            print(sample['img_gt_boxes'][0])
            
    
            if i_batch == 0:
                break
        
    if True: # 判断图片的x和保存的w,h是否一样
        file = pd.read_csv(csv_file)
        i = 7892
        name = file.iloc[i][0]
        size = file.iloc[i][1]
        print(name, size)
        path = 'resize/JPEGImages/'+name
        x = Image.open(path)
        print(x)

    
    