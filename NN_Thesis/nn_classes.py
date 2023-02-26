import torch

import torch.nn as nn
import torch.nn.functional as F
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
