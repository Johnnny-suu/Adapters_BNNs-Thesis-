'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def binarise_layer(layer,*args,**kwargs):
    #Takes in a module and binarises the forward pass
    class binary_layer(layer):
        def __init__(self,*args,**kwargs):
            #Recreate the layer properties
            super().__init__(*args,**kwargs)
            
            #We need the original forward function
            self.org_forward = super().forward
            self.forward = self.forward_WaB

            if hasattr(self,'weight'):
                self.org_weight_data = self.weight.data
                
            if hasattr(self,'bias'):
                if self.bias is not None:
                    self.org_bias_data = self.bias.data
                else:
                    self.forward = self.forward_noBias
            
        def forward_WaB(self,x):
            #Can replace with for loop over layer.parameters()?
            #Replace the weights and biases with binarisation and then return them to orig after
            
            self.org_weight_data = self.weight.data
            self.weight.data = torch.sign(self.weight.data)
        
            self.org_bias_data = self.bias.data
            self.bias.data = torch.sign(self.bias.data)
            out = self.org_forward(x)
#             print(self.org_forward)
            #Replace the weights with original weights
            self.weight.data = self.org_weight_data
            self.bias.data = self.org_bias_data
            return out
        
        def forward_noBias(self,x):
            self.org_weight_data = self.weight.data
            self.weight.data = torch.sign(self.weight.data)
            print(self.weight.data)
            out = self.org_forward(x)

            #Replace the weights with original weights
            self.weight.data = self.org_weight_data
            return out

    return binary_layer(*args,**kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,binarise = True):
        super(BasicBlock, self).__init__()

        self.binary = binarise
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)



        if binarise:
            self.conv1 = binarise_layer(nn.Conv2d,in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = binarise_layer(nn.Conv2d,planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,binarise = True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,binarise = binarise)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,binarise = binarise)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,binarise = binarise)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,binarise = binarise)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride,binarise):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,binarise))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

test()
