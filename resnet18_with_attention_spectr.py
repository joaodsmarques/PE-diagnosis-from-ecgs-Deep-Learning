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
                        #nn.Linear(in_features = 120, out_features = 2)
                        #nn.Softmax(dim = 1)
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


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm1d(out_channels*self.expansion, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.attention = nn.MultiheadAttention(512, 4, dropout = 0.2, batch_first = True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*64, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)    
        x = self.layer2(x)
 
        x = self.layer3(x)
     
        x = self.layer4(x)
       
        x = torch.permute(x,(0,2,1))
      
        x ,_ = self.attention(x,x,x)
       
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes*ResBlock.expansion,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet18_1d_with_att(num_classes, channels=12):
    return ResNet(Bottleneck, [2,2,2,2], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

