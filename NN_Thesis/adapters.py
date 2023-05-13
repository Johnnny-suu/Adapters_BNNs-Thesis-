import torch
import torch.nn as nn
import torch.nn.functional as F


class split_Conv2D(nn.Module):
    '''
    Splits the channels into S groups and performs Convolution2D on each group independently.

    No. of weights = C_in^2/S
    '''
    def __init__(self,in_channels:int,split:int,bias: bool = False ) -> None:
        super().__init__()
        if (in_channels % split)!= 0:
            raise ValueError(f'number of in channels must be devisible by split given')
        
        self.channels_per_split = in_channels//split
        self.split = split
        self.in_channels = in_channels

        self.groups = torch.arange(0,in_channels,1).reshape(split,self.channels_per_split)
        self.conv_layers = nn.ModuleList([nn.Conv2d(self.channels_per_split,self.channels_per_split,1,1,0,bias = bias) for _ in range(split)])

    def forward(self,x):
        '''
        Assumes input size of (B,C,H,W)
        '''
        out = torch.cat([ conv(x[:,group,:,:]) for conv,group in zip(self.conv_layers,self.groups)],dim = 1)
        return out



class adapterModule(nn.Module):
    '''
    Parent Class for adapters. Useful for adding general adapter functions
    Inherits from nn.Module so difference between is the 2 is small
    '''
    def __init__(self) -> None:
        super().__init__()
    def init_weight_zeros(self):
        #Set Parameters Weights to zero
        # Equivalent Run of not having Adapters inserted into network
        for p in self.parameters():
            nn.init.zeros_(p)



class identity_adapter(adapterModule):
    '''
    For debugging of trainer method
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        return x


class mini_bottleneck_adapter(adapterModule):
    def __init__(self,input_shape,num_adapters,downsample,nonlinearity:str = 'relu',bias:bool = True) -> None:
        super().__init__()
        #= (320,8,8)
        self.shape = input_shape
        input = prod(self.shape[1:])
        self.l_in = nn.Linear(input,downsample, bias)
        self.non_lin = getattr(F,nonlinearity)
        self.bn1 = nn.BatchNorm1d(downsample)
        self.l_out = nn.Linear(downsample,input,bias)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self,x):
        mini_x = x[:,0,:].clone() #Clone 8x8
        #Turn Shape into 1D Vector
        mini_x = nn.Sequential(nn.Flatten(),self.l_in)(mini_x)
        #Apply Non linearity
        mini_x = self.bn1(mini_x)
        mini_x = self.non_lin(mini_x)
        #Convert shape back to original shape
        mini_x =  nn.Sequential(self.l_out,nn.Unflatten(1,self.shape))(mini_x)

        x[:,0,:] += mini_x

        return x
        

class autoencoder_adapter(adapterModule):

    def __init__(self,input,downsample,nonlinearity = 'relu',bias:bool = True) -> None:
        super().__init__()
        self.l_in = nn.Linear(input,downsample, bias)
        self.non_lin = getattr(F,nonlinearity)
        self.bn1 = nn.BatchNorm1d(downsample)
        self.l_out = nn.Linear(downsample,input,bias)
        self.bn2 = nn.BatchNorm1d(input)
    def forward(self,x):
        out = self.l_in(x)
        out = self.bn1(out)
        out = self.non_lin(out)
        
        out = self.l_out(out)

        return F.hardtanh(self.bn2(x + out))




class bottleneck_adapter(adapterModule):
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

class conv_adapter(adapterModule):
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



class conv_channel_adapter(adapterModule):
    '''
    Reduces the number of channels and then upscales back to the original number of channels
    

    No of weights: C_in*C_mid*k*k + C_mid*C_in*k*k. Cheaper than regular conv adapter if C_mid < C_in

    '''
    def __init__(self,in_channels,down_channels,kernel = 1,stride = 1,padding = 0,bias:bool = False,nonlinearity:str = 'hardtanh'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,down_channels,kernel,stride,padding,bias = bias)
        self.conv2 = nn.Conv2d(down_channels,in_channels,kernel,stride,padding,bias = bias)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.non_lin = getattr(F,nonlinearity)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self,x):
        

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.non_lin(out) 
        out = self.conv2(out)
        out = self.bn2(out+x)

        return F.hardtanh(out)



class conv_channel_adapter2(adapterModule):
    '''
    Reduces the number of channels and then upscales back to the original number of channels
    

    No of weights: C_in*C_mid*k*k + C_mid*C_in*k*k. Cheaper than regular conv adapter if C_mid < C_in

    This has the first bn at the downsample before the nonlinerarity 
    '''
    def __init__(self,in_channels,down_channels,kernel = 1,stride = 1,padding = 0,bias:bool = False,nonlinearity:str = 'hardtanh'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,down_channels,kernel,stride,padding,bias = bias)
        self.conv2 = nn.Conv2d(down_channels,in_channels,kernel,stride,padding,bias = bias)
        
        self.bn1 = nn.BatchNorm2d(down_channels)
        self.non_lin = getattr(F,nonlinearity)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.non_lin(out) 
        out = self.conv2(out)
        out = self.bn2(out+x)

        return F.hardtanh(out)



class conv_channel_adapter3(adapterModule):
    '''
    Splits the features maps into S groups/ For example if the input channel is 80 and we want to split the channels into 10 groups then each group
    contains 8 channels. Each group is processed independently like it's own Convolutional layer.

    No of weights: C_in^2 / S.

    S should be an integer between 1 and C_in (inclusive). setting S = Cin would be processing each feature map independently while S = 1 would
    include all channels in one group so would be standard, 2D Convolution

    This has the first bn at the downsample before the nonlinerarity 
    '''
    def __init__(self,in_channels,kernel = 1,stride = 1,padding = 0,groups = None,bias:bool = False,nonlinearity:str = 'relu'):
        super().__init__()
        
        if groups is None:
            groups = in_channels
        self.conv = nn.Conv2d(in_channels,in_channels,kernel,stride,padding,groups = groups,bias = False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.non_lin = getattr(F,nonlinearity)
    def forward(self,x):
        
        out = self.bn1(x)
        out = self.conv(out)
        out = self.bn2(out+x)

        return F.hardtanh(out)








class conv_bottleneck_adapter(adapterModule):
    '''
    More general convulutional adapter. Can either downsample channels only, img dimensions only or both choices
    '''
    def __init__(self,in_channels,down_channels,kernel = 1,stride = 1,padding = 0,bias:bool = False,nonlinearity:str = 'relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,down_channels,kernel,stride,padding,bias = bias)
        self.bn1 = nn.BatchNorm2d(down_channels)
        self.non_lin = getattr(F,nonlinearity)
        self.deconv1 = nn.ConvTranspose2d(down_channels,in_channels,kernel,stride,padding,bias = bias)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.non_lin(out)
        out = self.deconv1(out)
        out = self.bn2(out+x)
        return F.hardtanh(out)


import random

class partial_conv_adapter(conv_adapter):
    def __init__(self,in_channels:int,internal_channels:int,kernel = 1,stride = 1,padding = 0,bias:bool = False,nonlinearity:str = 'hardtanh'):
        super().__init__(in_channels,kernel,stride,padding,bias,nonlinearity)
        self.in_channels = in_channels
        self.internal_channels = internal_channels
        if internal_channels > self.in_channels:
            raise ValueError('number of internal channels must be less than or equal to number of input channels')

        self.ToSample = self.policy()

    def policy(self):
        return random.sample( list(range(self.in_channels)), self.internal_channels )
        
if __name__ == '__main__':
    conv = conv_adapter(2)
    conv.init_weight_zeros()

    for p in conv.parameters():
        print(p)