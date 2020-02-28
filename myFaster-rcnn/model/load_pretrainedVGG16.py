# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:40:44 2020

@author: tengjunwan
"""

from torchvision import models
import torch

def load_vgg16(path='pretrained_model/checkpoints/vgg16-397923af.pth'):
    vgg16 = models.vgg16()
    vgg16.load_state_dict(torch.load(path))
    return vgg16
    
    
if __name__ == "__main__":
    path = 'pretrained_model/checkpoints/vgg16-397923af.pth'
    vgg16 = load_vgg16(path)