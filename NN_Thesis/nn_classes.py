import torch

import torch.nn as nn
import torch.nn.functional as F
if __name__ == '__main__':
    from models import *
    from torch.utils.data import DataLoader
else:
    from .models import *

class SimpleCNN(nn.Module):
    '''
    Basic Neural Network from Pytorch Tutorial, Good for testing adapters
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
#         self.type = self.conv1.weight.dtype
        
    def forward(self,x):
#         x = x.type(self.conv1.weight.dtype)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
#         x = F.softmax(x,dim = -1 )
        return x


class resnet18_adapt(ResNet_cifar10):

    def __init__(self,num_classes,state_dict = None,freeze_pre_weights = True):
        super().__init__(num_classes= num_classes)
        if state_dict is not None:
            self.load_state_dict(state_dict)
            if freeze_pre_weights:
                self.freeze()
                print('Pretrained weights have been frozen')
            else:
                print('Pretrained weights have not been frozen')
        #Store the adapters in a dict and keep ID for repetive ones
        self.adapter_dict = nn.ModuleDict()
        self.adapter_id = 0
    def freeze(self):
        #Set all layers to eval() mode and freeze parameters
        for c in self.children():
            c.eval()
        for p in self.parameters():
            p.requires_grad = False
    def unfreeze(self):
        for c in self.children():
            c.train()
        for p in self.parameters():
            p.requires_grad = True
    def add_adapter(self,after,adapter):
        '''
        Add adapter to a model 'after' a layer:

        After can be a string or a list of strings of layer name(s). If after is a list of strings, adapter must also be a list of the same length

        You can add multiple adapters after the same layer by repeating a layer e.g. add_adapter([layer1,layer1],[ad_nn1,ad_nn2])

        inputs:
            after: type(str,list,tuple) if "after" is str of layer name to add adapter afterwards or list/tuple of n layers to add n adapters to after
            adapter: nn.Module layer to add to the model. 
        '''

        #Attach adapter to model using NN.sequentia
        def addAdapter(layer,adapter_):

            #Check if layer to append after actually exists
            if hasattr(self,layer):
                model_layer = getattr(self,layer)
                #Add to ModuleDict:
                key = f'{layer},{type(adapter_).__name__},{self.adapter_id}'
                self.adapter_dict[key] = adapter_
                self.adapter_id += 1
                if isinstance(model_layer,nn.Sequential):
                    model_layer.append(self.adapter_dict[key])
                else:
                    model_layer = nn.Sequential(model_layer,self.adapter_dict[key])
            else:
                raise ValueError(f'layer with name "{layer}" does not exist!')

        if isinstance(after,str):
            addAdapter(after,adapter)

        elif isinstance(after,(list,tuple) ) and isinstance(adapter,(list,tuple)):

            assert len(after) == len(adapter)
            
            for layer,adpter in zip(after,adapter):
                addAdapter(layer,adpter)
        else:
            #Raise if after and adapter are not both list like
            raise TypeError('Please ensure that both "after" and "adapter" are both list like and the same length')

class resnet18_adapt_Cifar100(ResNet_Cifar100):

    def __init__(self,num_classes,state_dict = None,freeze_pre_weights = True,**kwargs):
        super().__init__(num_classes= num_classes,**kwargs)
        if state_dict is not None:
            self.load_state_dict(state_dict)
            if freeze_pre_weights:
                self.freeze()
                print('Pretrained weights have been frozen')
            else:
                print('Pretrained weights have not been frozen')
        #Store the adapters in a dict and keep ID for repetive ones
        self.adapter_dict = nn.ModuleDict()
        self.adapter_id = 0
    def freeze(self):
        #Set all layers to eval() mode and freeze parameters
        for c in self.children():
            c.eval()
        for p in self.parameters():
            p.requires_grad = False
    def unfreeze(self):
        for c in self.children():
            c.train()
        for p in self.parameters():
            p.requires_grad = True
    def add_adapter(self,after,adapter):
        '''
        Add adapter to a model 'after' a layer:

        After can be a string or a list of strings of layer name(s). If after is a list of strings, adapter must also be a list of the same length

        You can add multiple adapters after the same layer by repeating a layer e.g. add_adapter([layer1,layer1],[ad_nn1,ad_nn2])

        inputs:
            after: type(str,list,tuple) if "after" is str of layer name to add adapter afterwards or list/tuple of n layers to add n adapters to after
            adapter: nn.Module layer to add to the model. 
        '''

        #Attach adapter to model using NN.sequentia
        def addAdapter(layer,adapter_):

            #Check if layer to append after actually exists
            if hasattr(self,layer):
                model_layer = getattr(self,layer)
                #Add to ModuleDict:
                key = f'{layer},{type(adapter_).__name__},{self.adapter_id}'
                self.adapter_dict[key] = adapter_
                self.adapter_id += 1
                if isinstance(model_layer,nn.Sequential):
                    model_layer.append(self.adapter_dict[key])
                else:
                    model_layer = nn.Sequential(model_layer,self.adapter_dict[key])
            else:
                raise ValueError(f'layer with name "{layer}" does not exist!')

        if isinstance(after,str):
            addAdapter(after,adapter)

        elif isinstance(after,(list,tuple) ) and isinstance(adapter,(list,tuple)):

            assert len(after) == len(adapter)
            
            for layer,adpter in zip(after,adapter):
                addAdapter(layer,adpter)
        else:
            #Raise if after and adapter are not both list like
            raise TypeError('Please ensure that both "after" and "adapter" are both list like and the same length')


def compare_weights(main_model:nn.Module,*models:nn.Module):
    '''
    For checking if weight are the same
    '''
    for model in models:
        for name,mod in model.named_children():
            # print(name)
            if hasattr(main_model,name):
                #Compare weights of modules
                main_mod = getattr(main_model,name)
                if hasattr(mod,'weight') and hasattr(main_mod,'weight'):
                    is_same = torch.all(mod.weight == main_mod.weight)
                    if is_same is False:
                        print(f'Child Module {name} weights do not match with models')
            else:
                print(f'Main model does have child {name}')



class thinBlock(nn.Module):
    def __init__(self,in_channels,thin_channel,out_channels,groups = [1,1]):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels,thin_channel,kernel_size = 1,groups= groups[0])
        self.bn1 = nn.BatchNorm2d(thin_channel)
        
        #Depthwise Convolution
        self.conv2 = nn.Conv2d(thin_channel,thin_channel,groups = thin_channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(thin_channel)

        #PointWise Convolution
        self.conv3 = nn.Conv2d(thin_channel,out_channels,kernel_size=1,groups= groups[1 ])
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        out = nn.Sequential(self.conv1,self.bn1,nn.ReLU())(x)
        out = nn.Sequential(self.conv2,self.bn2,nn.ReLU())(out)
        return nn.Sequential(self.conv3,self.bn3)(out)

class thinBlock2(nn.Module):
    def __init__(self,in_channels,out_channels,conv1_group = None,conv2_group =None,bias = False):
        super().__init__()
        
        if conv1_group is None:
            conv1_group = in_channels
        if conv2_group is None:
            conv2_group = min(in_channels,out_channels)

        #Depthwise Convolution
        self.conv2 = nn.Conv2d(in_channels,in_channels,groups = conv1_group,kernel_size=3,stride=1,padding=1,bias=bias)
        self.bn2 = nn.BatchNorm2d(in_channels)

        #PointWise Convolution
        self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1,groups=  conv2_group,bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        out = nn.Sequential(self.conv2,self.bn2,nn.ReLU())(x) + x
        return nn.Sequential(self.conv3,self.bn3)(out)

class thinBlock3(nn.Module):
    def __init__(self,in_channels,out_channels,bias = False):
        super().__init__()
        
        #Depthwise Convolution
        self.conv2 = nn.Conv2d(in_channels,in_channels,groups = in_channels,kernel_size=3,stride=1,padding=1,bias=bias)
        self.bn2 = nn.BatchNorm2d(in_channels)

        #PointWise Convolution
        self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        out = nn.Sequential(self.conv2,self.bn2,nn.ReLU())(x) + x
        return nn.Sequential(self.conv3,self.bn3)(out)




class crazy_adaptNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x:list):
        f1,f2,f3 = x
        
        out1 =nn.Flatten()(nn.AdaptiveAvgPool2d(1)(f1))
        out2 = nn.Flatten()(nn.AdaptiveAvgPool2d(1)(f2))
        out3 = nn.Flatten()(nn.AdaptiveAvgPool2d(1)(f3))
        #Vector size is 80 + 160 + 320 = 560
        return torch.cat([out1,out2,out3],dim = -1)
        


class uniAdapt_Net(nn.Module):
    def __init__(self,input_feature_channels:list,out_channels:list,block,pooling = 'avg',*args,**kwargs) -> None:
        super().__init__()
        self.layer1 = block(input_feature_channels[0],out_channels[1],*args,**kwargs)
        self.layer2 = block(input_feature_channels[1],out_channels[2],*args,**kwargs)
        self.layer3 = block(input_feature_channels[2],out_channels[2],*args,**kwargs)

        if pooling == 'avg':
            self.pool1 = nn.AdaptiveAvgPool2d(16)
            self.pool2 = nn.AdaptiveAvgPool2d(8)
        elif pooling == 'max':
            self.pool1 = nn.AdaptiveMaxPool2d(16)
            self.pool2 = nn.AdaptiveMaxPool2d(8)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
    def forward(self,x:list):
        f1,f2,f3 = x
        # f1 80,32,32
        # f2 160,16,16
        # f3 320,8,8
        out = self.layer1(f1)
        out = self.pool1(out)
        out = self.layer2(f2 + out)
        out = self.pool2(out)
        out = self.layer3(f3 + out)
        out = self.pool3(out)
        return out        
    



class uniAdapt_Net_React (nn.Module):
    def __init__(self,input_feature_channels:list,out_channels:list,block,pooling = 'avg',*args) -> None:
        super().__init__()
        self.layer1 = block(input_feature_channels[0],out_channels[1],*args)
        self.layer2 = block(input_feature_channels[1],out_channels[2],*args)
        self.layer3 = block(input_feature_channels[2],out_channels[3],*args)
        self.layer4 = block(input_feature_channels[3],out_channels[3],*args)
        if pooling == 'avg':
            self.pool1 = nn.AdaptiveAvgPool2d(28)
            self.pool2 = nn.AdaptiveAvgPool2d(14)
            self.pool3 = nn.AdaptiveAvgPool2d(7)
        elif pooling == 'max':
            self.pool1 = nn.AdaptiveMaxPool2d(28)
            self.pool2 = nn.AdaptiveMaxPool2d(14)
            self.pool3 = nn.AdaptiveMaxPool2d(7)

        self.pool4 = nn.AdaptiveAvgPool2d(1)
    def forward(self,x:list):
        f1,f2,f3,f4 = x
        # torch.Size([1, 64, 56, 56])
        # torch.Size([1, 128, 28, 28])
        # torch.Size([1, 256, 14, 14])
        # torch.Size([1, 512, 7, 7])
        out = self.layer1(f1)
        out = self.pool1(out)
        out = self.layer2(f2 + out)
        out = self.pool2(out)
        out = self.layer3(f3 + out)
        out = self.pool3(out)
        out = self.layer4(f4+out)
        out = self.pool4(out)
        return nn.Flatten()(out)        
    



class test_adaptnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x:list):
        bs = x[0].shape[0]
        return torch.zeros((bs,1)).cuda()

class BNN_UniAdapt_encoders(nn.Module):
    def __init__(self,in_channels,down_channels,kernel_size,stride=1,padding = 0,groups = 1):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Conv2d(in_channel,down_channel,kernel_size,stride,padding,groups = groups) for in_channel,down_channel in zip(in_channels,down_channels) ])
        self.decoders = nn.ModuleList([nn.Conv2d(down_channel,in_channel,kernel_size,stride,padding,groups = groups) for in_channel,down_channel in zip(in_channels,down_channels) ])    
        self.non_lin = nn.ReLU()
    def encode(self,x):
        return [self.non_lin(encode(f)) for encode,f in zip(self.encoders,x)]
    def decode(self,z):
        return [decode(f) for decode,f in zip(self.decoders,z)]
    
    def forward(self,x:list,train = False):
        z = self.encode(x)
        if train:
            return self.decode(z)
        else:
            return z 
    
    
class BNN_Resnet_UniAdapt(resnet18_adapt):
    def __init__(self,num_classes,state_dict = None,freeze_pre_weights = True):
        super().__init__(num_classes,state_dict,freeze_pre_weights)
        self.uniAdapt = False
        self.head_bn = nn.BatchNorm1d(320 + 160)
        self.head = nn.Linear(320+160,num_classes)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    def add_UniAdapter(self, adapter_net):
        self.uniAdaptNet = adapter_net
    
    def add_Encoders(self,encoder_net):
        self.encoder_net = encoder_net
    def forward(self, x):
        if self.uniAdapt:
            with torch.no_grad():
                #Regular Backbone Forward Pass but with no grad
                x = self.conv1(x)
                x = self.maxpool(x)
                x = self.bn1(x)
                x = self.tanh1(x)
                f1 = self.layer1(x)
                
                f2 = self.layer2(f1)

                f3 = self.layer3(f2)
                backbone_out = self.avgpool(f3)
                backbone_out = backbone_out.view(backbone_out.size(0), -1)
                backbone_out = self.bn2(backbone_out)
                backbone_out = self.tanh2(backbone_out)
            
                #Encode to latent space Z
                encode = self.encoder_net([f1,f2,f3])

            #Return the input features maps and decoded result 
            UniAdapter_out = self.uniAdaptNet(encode)
            
            #Assume that a 1D output of size N,100 is given by UniAdapter
            concat_layer = torch.cat((backbone_out,UniAdapter_out),dim = -1)
            
            lin_out   =self.head(concat_layer)

            return self.head_bn(lin_out)

        #if False do normal pass through backbone
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        f1 = self.layer1(x)
        
        f2 = self.layer2(f1)

        f3 = self.layer3(f2)
        
        x = self.layer4(f3)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)
        return x

    
    def train_encoders(self,epochs,trainloader,lr = 0.001):

        optimizer = torch.optim.Adam(self.encoder_net.parameters(),lr=lr)
        optimizer.zero_grad()
        for epoch in range(epochs):
            running_loss = [0,0,0]
            for data in trainloader:
                #Only need Image no need for label
                x = data[0].to(self.device)
                # x = data.to(self.device) 
                with torch.no_grad():
                    #Regular Backbone Forward Pass but with no grad
                    x = self.conv1(x)
                    x = self.maxpool(x)
                    x = self.bn1(x)
                    x = self.tanh1(x)
                    f1 = self.layer1(x)
                    
                    f2 = self.layer2(f1)
                    f3 = self.layer3(f2)


                decodes = self.encoder_net([f1,f2,f3],train= True)
                losses = ([(decode - f).pow(2).mean() for decode,f in zip(decodes,[f1,f2,f3])])
                loss = sum(losses)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss = [r + l for r,l in zip(running_loss,losses)]
            print([r/len(trainloader) for r in running_loss])





class BNN_Resnet_Cifar100_UniAdapt(resnet18_adapt_Cifar100):
    def __init__(self,num_classes,state_dict = None,freeze_pre_weights = True,**kwargs):
        super().__init__(num_classes,state_dict,freeze_pre_weights,**kwargs)
        self.uniAdapt = False
        self.head_bn = nn.BatchNorm1d(320 + 160)
        self.head = nn.Linear(320+160,num_classes)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    def add_UniAdapter(self, adapter_net):
        self.uniAdaptNet = adapter_net
    
    def add_Encoders(self,encoder_net):
        self.encoder_net = encoder_net
    def forward(self, x):
        if self.uniAdapt:
            with torch.no_grad():
                #Regular Backbone Forward Pass but with no grad
                x = self.conv1(x)
                x = self.maxpool(x)
                x = self.bn1(x)
                x = self.tanh1(x)
                f1 = self.layer1(x)
                
                f2 = self.layer2(f1)

                f3 = self.layer3(f2)
                backbone_out = self.avgpool(f3)
                backbone_out = backbone_out.view(backbone_out.size(0), -1)
                backbone_out = self.bn2(backbone_out)
                backbone_out = self.tanh2(backbone_out)
            
                #Encode to latent space Z
                encode = self.encoder_net([f1,f2,f3])

            #Return the input features maps and decoded result 
            UniAdapter_out = self.uniAdaptNet(encode)
            
            #Assume that a 1D output of size N,100 is given by UniAdapter
            concat_layer = torch.cat((backbone_out,UniAdapter_out),dim = -1)
            
            lin_out   =self.head(concat_layer)

            return self.head_bn(lin_out)

        #if False do normal pass through backbone
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        f1 = self.layer1(x)
        
        f2 = self.layer2(f1)

        f3 = self.layer3(f2)
        
        x = self.layer4(f3)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)
        return x




if __name__ == '__main__':
    x = torch.rand((4,3,32,32))

    y = [torch.rand((4,80,32,32)),torch.rand((4,160,16,16)),torch.rand((4,320,8,8))]

    n1 = BNN_Resnet_UniAdapt(10)
    
    n2 = BNN_UniAdapt_encoders([80,160,320],[40,80,160],1)
    print([out.shape for out in n2(y)]) 
    n3 =  uniAdapt_Net([40,80,160],[40,80,160],block = thinBlock2)
    n3 = nn.Sequential(n3,nn.Flatten())
    n1.add_Encoders(n2)
    n1.add_UniAdapter(n3)
    n1.uniAdapt = True


    # print(n1(x))
    

    trainloader = DataLoader(torch.rand((10,3,32,32)),batch_size=10)
    n1 = n1.to(device = 'cuda:0')

    # for name,c in n1.named_children():
    #     print(name)
    
    # test_n = resnet18_adapt(10)
    # test_n.to('cuda:0')
    # print(test_n(x.cuda()))

    # print(n1(x.cuda()))

    n1.train_encoders(1,trainloader)