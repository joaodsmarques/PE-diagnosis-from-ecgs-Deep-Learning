# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:05:27 2023

@author: Utilizador
"""
import torch
import torch.nn as nn
import sys
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
#Encoder class
class ECGencoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dropout):
        super(ECGencoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first = True, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)


    def forward(self, x):
        
        #Batch first true means (Batch, seq length, features)
        out = self.transformer_encoder(x)

        return x
    
    '''
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    '''
#For trnsformers and encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=2):
        super(ResNet, self).__init__()

        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=17, stride=2, padding=8, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        #Transformer encoder
        self.encoder = ECGencoder(d_model = 512, n_heads = 4, num_layers = 3, dropout = 0.2)
        self.pos_enc = PositionalEncoding(d_model = 512)
        
        #self.attention = nn.MultiheadAttention(512, 4, batch_first = True, dropout = 0.3)
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                                nn.Linear(512*64, num_classes),
                                nn.Dropout(0.3),
                                )
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x) 

        x = self.layer2(x)
 
        x = self.layer3(x)
     
        x = self.layer4(x)
       
        #input is (N L EQ) where channels is Eq. that s why permuted is needed - for rnns and multihead
        #So it gets batch, signal, channels

        #Encoder part
        #Get seq length, batch_size, features
        x = torch.permute(x,(2,0,1))
        x = self.pos_enc(x)
        #Batch, seq_length,features
        x = torch.permute(x,(1,0,2))

        x = self.encoder(x)
        out = torch.flatten(x, 1)
        
        out = self.fc(out)
        
        return out
        
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

        
        
def ResNet18_with_Encoder(num_classes, channels=12):
    return ResNet(Bottleneck, [2,2,2,2], num_classes, channels)