# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:22:35 2020

@author: tengjunwan
"""
import torch.nn as nn
from model.load_pretrainedVGG16 import load_vgg16
from model.region_proposal_network import RegionProposalNetwork
import torch

"""该模块是组合了vgg16和rpn，forward()输出为rpn的输出，rpnlocs(1,#anchors,4)和
rpnscores(1,#anchors,2),以及anchors(这主要是为了下一步的roi proposal)
"""
def decomVgg16(path):
    """切割并freez vgg16，返回一个nn.Sequential和它的classifier(两层fc,relu激活)
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
    

class Vgg16ToRPN(nn.Module):
    def __init__(self, path):
        super(Vgg16ToRPN, self).__init__()
        self.extractor, _ = decomVgg16(path)
        self.rpn = RegionProposalNetwork()
    
    def forward(self, x):
        # 输入x是(batchsize=1,channel=3,width=W,height=H)
        h = self.extractor(x)  # h是(1,channel=512,w,h)
        rpn_locs, rpn_scores, anchors = self.rpn(h)
        return rpn_locs, rpn_scores, anchors
    
    
    
    
if __name__ == "__main__":
    path = 'pretrained_model/checkpoints/vgg16-397923af.pth'
    vgg16_to_rpn = Vgg16ToRPN(path)
    x = torch.rand((1, 3, 600, 850))  # 第一张图片
    rpn_locs, rpn_scores, anchors = vgg16_to_rpn(x)  
    print("rpn_locs:", rpn_locs.shape)  # [1, 17649, 4]
    print("rpn_scores:", rpn_scores.shape)  # [1, 17649, 2]
    print("anchors:", anchors.shape)  # [17649, 4]
    _, classifier = decomVgg16(path) # 看下源代码中classifier是什么东西
