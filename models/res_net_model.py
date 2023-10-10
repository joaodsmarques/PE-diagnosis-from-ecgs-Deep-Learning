# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:44:38 2022

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



###############################
#Model used as baseline

## Variables

#Class to define the residual layers
#channels define the dimension of each channel
class ResidualBlock(torch.nn.Module):
    """ Helper Class"""

    def __init__(self, channels, downsample):
        
        super(ResidualBlock, self).__init__()
        
        self.block = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=channels[0],
                                out_channels=channels[1],
                                kernel_size= 16,
                                padding= 'same'
                                ),
                torch.nn.BatchNorm1d(channels[1]),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.2),
                torch.nn.Conv1d(in_channels=channels[1],
                                out_channels=channels[2],
                                kernel_size=17,
                                stride= downsample,
                                padding= 8
                                ),
        )

        self.shortcut = torch.nn.Sequential(
                torch.nn.MaxPool1d(kernel_size = downsample),
                torch.nn.Conv1d(in_channels=channels[0],
                                out_channels=channels[2],
                                kernel_size=1,
                                bias = False,
                                padding = 'same')
                )
                
        self.exit_action = torch.nn.Sequential(
                torch.nn.BatchNorm1d(channels[2]),
                torch.nn.ReLU(inplace = True),
                torch.nn.Dropout(0.2)
            )
        
    #Try to make the first block being 0
    def forward(self, skip, x):
        
        #Block of data 
        main_connection = self.block(x)
        if skip is not None:
          #Skip connection operations
          shortcut = self.shortcut(skip)
          #print(block.size(), shortcut.size())
          next_skip = shortcut + main_connection
        else:
          '''
            think if should be better sum
            the x entry when in the first layer
          ''' 
          next_skip = main_connection
          
        #x = nn.ReLU(next_skip)
        main_connection = self.exit_action(next_skip)

        return (next_skip, main_connection)
    
    
class Special_Net(nn.Module):
    
    def __init__(self):
        
        super(Special_Net,self).__init__()
        
        # self.init_weights
        
        self.entry_block = nn.Sequential( 
                                    nn.Conv1d(in_channels = 12, out_channels = 64, kernel_size = 16, padding = 'same'),
                                    nn.BatchNorm1d(64), 
                                    nn.ReLU(),
                                    )
        self.rb1 = ResidualBlock([64,128,128],4)
        self.rb2 = ResidualBlock([128,192,192],4)
        self.rb3 = ResidualBlock([192,256,256],4)
        self.rb4 = ResidualBlock([256,320,320],2)
        self.rb5 = ResidualBlock([320,320,384],2)
        
        
        self.output = nn.Sequential(
                                    nn.Flatten(start_dim = 1),
                                    nn.Linear(in_features = 16*384, out_features = 2),
                                    nn.Dropout(0.5),
                                    #nn.Linear(in_features = 200, out_features = 2)
                                    #nn.Dropout(0.2),                                    
                                    #nn.Linear(in_features = 64, out_features = 2)
                                    
                                    #nn.Sigmoid()
            
            )
    #Network forward execution
    def forward(self, x):
        forward_result = self.entry_block(x)
        shortcut , forward_result = self.rb1.forward(None, forward_result)
        shortcut , forward_result = self.rb2.forward(shortcut, forward_result)
        shortcut , forward_result = self.rb3.forward(shortcut, forward_result)
        shortcut , forward_result = self.rb4.forward(shortcut, forward_result)
        shortcut , forward_result = self.rb5.forward(shortcut, forward_result)
        
        
        #print(forward_result.size())
        
        forward_result = self.output(forward_result)
        
        return forward_result
    
    
'''   
model = torchvision.models.resnet18(pretrained = False)

print(model)
'''
