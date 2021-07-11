import torch
from torch import nn as nn
import math
import logging
from mdset import LunaDataset
from torch.utils.data import DataLoader



class LunaBlock(nn.Module):
    def __init__(self, in_chanels, conv_chanels, ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chanels, conv_chanels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(conv_chanels, conv_chanels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool3d(2,2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        return self.maxpooling(out)

class LunaModel(nn.Module):
    def __init__(self, in_chanels = 1, conv_chanels = 8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_chanels, conv_chanels)
        self.block2 = LunaBlock(conv_chanels, conv_chanels * 2)
        self.block3 = LunaBlock(conv_chanels * 2, conv_chanels * 4)
        self.block4 = LunaBlock(conv_chanels * 4, conv_chanels * 8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear, 
                nn.Conv2d, 
                nn.Conv3d, 
                nn.ConvTranspose2d, 
                nn.ConvTranspose3d
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        model_out = self.tail_batchnorm(x)
        model_out = self.block1(model_out)
        model_out = self.block2(model_out)
        model_out = self.block3(model_out)
        model_out = self.block4(model_out)
        
        out_flat = model_out.view(model_out.shape[0], -1)
        
        linear_out = self.head_linear(out_flat)
        return linear_out, self.head_softmax(linear_out)

def computeBackLoss(batch_ndx, batch_tup, batch_size, metrics_g):
    loss_func = nn.CrossEntropyLoss(reduce='none')

if __name__ == '__main__':
    dt = LunaDataset()
    dl = DataLoader(dt, batch_size=2)
    loss_func = nn.CrossEntropyLoss(reduce=False)
    model = LunaModel(1, 8)
    for i in dl:
        data, label, _, _ = i
        print(label)
        out, probability = model(data)
        loss = loss_func(out, label[:, 1])
        break
    # input = torch.stack([dt[0][0], dt[1][0]])
    # print(input.shape)
    # block = LunaModel(1, 8)
    # conv3d = nn.Conv3d(1,1,3,bias=True)
    # output1, output2 = block(input)
