# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:33:28 2020

@author: tengjunwan
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def draw_predict(dir_path, img_name, bboxes, labels, scores):
    """
    参数:
        dir_path(str):
        img_name(str)
        bboxes(numpy):
        labels(numpy):
        scores(numpy):
    """
    #label=0表示'person', label=19表示'tvmonitor'
    classes = ['person',
           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
           ]
    
#    # 中文还是不行，乱码
#    classes = ['人',
#           '鸟', '猫', '牛', '狗', '马', '羊',
#           '飞机', '自行车', '船', '公交车', '轿车', '摩托车', '火车',
#           '瓶子', '椅子', '餐桌', '盆栽', '沙发', '电视'
#           ]
    file_path = dir_path + img_name
    im = np.array(Image.open(file_path))
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # 画图片
    ax.imshow(im)
    # 标记object detection
    for i in range(len(bboxes)):
        xmin = bboxes[i][0]
        ymin = bboxes[i][1]
        width = bboxes[i][2] - bboxes[i][0]
        height = bboxes[i][3] - bboxes[i][1]
        text = classes[labels[i]] + ":" + "%d"%(scores[i] * 100)+ "%"
        # 画bboxes
        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        # 标记label以及score
        ax.text(xmin, ymin, text, fontsize=8,
                bbox=dict(facecolor='green', alpha=1))
    
    plt.savefig("show_result/"+img_name+".png")
    plt.clf()