from math import prod
import torch.nn as nn
import torch.nn.functional as F


class bottleneck_adapter(nn.Module):
    '''
    adapted from: http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf and modified for CNN  
     Takes in the input shape for 1 tensor e.g for a Tensor shape of (N,C,H,W) set input shape to (C,H,W)
    Note that for CNN's fully connected layers can be very expensive so use with care
     A hardtanh is applied to the end to be inline with Binerized Neural networks

    inputs:\n
    input_shape = (tuple) of tensor with shape  (N,C,H,W,...) the first dimension is assummed to be the mini batch number

    downsample = (int) the number of variables to bottleneck. Note that no check is made to ensure downsample > # input nuerons

    nonlinearity = (str) the non-linearity to perform in the downsampled latent space. Must match a function from torch.functional

    bias = (bool) set to True to add Bias
    '''
    def __init__(self,input_shape,downsample:int,nonlinearity:str = 'relu',bias:bool = True) -> None:
        super().__init__()
        self.shape = input_shape
        input = prod(self.shape)
        self.l_in = nn.Linear(input,downsample, bias)
        self.non_lin = getattr(F,nonlinearity)
        self.bn1 = nn.BatchNorm1d(downsample)
        self.l_out = nn.Linear(downsample,input,bias)
        self.bn2 = nn.BatchNorm2d(self.shape[0])

    def forward(self,x):
        # print(x.shape)

        #Turn Shape into 1D Vector
        out = nn.Sequential(nn.Flatten(),self.l_in)(x)
        #Apply Non linearity
        out = self.bn1(out)
        out = self.non_lin(out)
        #Convert shape back to original shape
        out =  nn.Sequential(self.l_out,nn.Unflatten(1,self.shape))(out)
        
        return F.hardtanh(self.bn2(out+x))

class conv_adapter(nn.Module):
    '''
    No bottle neck
    '''
    def __init__(self,in_channels,kernel = 1,stride = 1,padding = 0,bias:bool = False,nonlinearity:str = 'hardtanh'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel,stride,padding,bias = bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.non_lin = getattr(F,nonlinearity)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self,x):
        
        out = nn.Sequential(self.bn1,self.conv1)(x)

        out = self.bn2(out+x)

        return self.non_lin(out)

class conv_bottleneck_adapter(nn.Module):
    def __init__(self,in_channels,down_channels,kernel = 1,stride = 1,padding = 0,bias:bool = False,nonlinearity:str = 'relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,down_channels,kernel,stride,padding,bias = bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.non_lin = getattr(F,nonlinearity)
        self.deconv1 = nn.ConvTranspose2d(down_channels,in_channels,kernel,stride,padding,bias = bias)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self,x):
        
        out = nn.Sequential(self.bn1,self.conv1)(x)
        out = self.non_lin(out)
        out = self.deconv1(out)
        out = self.bn2(out+x)
        return F.hardtanh(out)
