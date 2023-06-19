# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:24:10 2022

@author: Utilizador
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
from PIL import Image
import torchvision


class Resnet18_with_attention(nn.Module):
    def __init__(self, input_model):
        super(Resnet18_with_attention, self).__init__()
        self.features = nn.Sequential(*list(input_model.children()))[:-3]
        self.attention = nn.MultiheadAttention(512, 4 , bias = False, dropout = 0.1, batch_first = True)
        
        self.output = nn.Sequential(
        
                        nn.Linear(in_features = 512*8, out_features = 2),
                        nn.Dropout(0.3),
                        )
        #self.classifier = nn.Linear(512*512, num_classes)
        #self.avg_pool = nn.AdaptiveAvgPool2d((7,7))
    
    def forward(self, x):
        #print(self.features)
        x = self.features(x)
      
        
        #x = torch.flatten(x, 1)
        #Match required column order for attention
        
        x= torch.flatten(x,2)
        
        x = torch.permute(x,(0,2,1))
        #print(x.shape)
        
        x,_ = self.attention(x,x,x)        
        x = torch.flatten(x, 1)
       
        x = self.output(x)
       
        return x

def get_resnet_model():
    model = torchvision.models.resnet18(pretrained = False)
    #model = torchvision.models.resnet50(pretrained = False)
    #print(model)
    #sys.exit()
    
    model.fc = nn.Linear(512,120)
    model.fc2 = nn.Linear(120, 2)
    #model.final_layer = nn.LogSoftmax(dim = 1)
    
    model.conv1 = nn.Conv2d(12, 64, kernel_size= 7, stride= 2, padding = 3, bias=False)
  
    for param in model.parameters():
        param.requires_grad = True
    
    model = Resnet18_with_attention(model)
    #print(model)
    return model



########################################################
########################################################
########################################################
########################################################


