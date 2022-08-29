import torch
import torchvision
from torchvision.transforms import transforms
import numpy as np
import os
from PIL import Image
import random

from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


import time
import torch.optim as optim

#For Supervised Training Should help with not having to goddamn run this code all the time just have a loop changing the
#Hyperparameters
class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)


class Trainer():
    def __init__(self,model = None,name = 'Model_1',classes = None,seed = None,regime = None,binarise = False):
        
        self.name = name
        self.model = model
        self.binarise = binarise
        
        self.total_time = 0
        self.best_model_accuracy = 0
        self.best_loss = float('inf')
        self.cwd = os.getcwd()
        self.epoch_losses = None

        #Default Training Parameters
        self.lr = 0.1
        self.batch_size = 16
        self.epochs = 1
        self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss() 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,200)
        
        self.seed = seed
        
        #A dictionary where key is the epoch and value are the settings to apply
        self.regime = regime 

        self.settings = [self.lr,
        self.batch_size,
        self.epochs,
        self.device ,
        self.criterion, 
        self.optimizer, 
        self.seed
        ]
        #Datasets
        self.classes = classes
        
        #Flags for saving
        self.saveNet = True
        self.saveData = True

    def train(self,trainloader,testloader):
        
        print('start')
        self.print_config()

        if self.seed is not None:
            torch.manual_seed(self.seed)
        #Store Values as a tuple (loss,accuracy)
        self.model.to(device = self.device)
        self.epoch_losses = [(self.train_epoch(epoch,trainloader),self.test(testloader)) for epoch in range(self.epochs)]

        if self.saveData:
            pass #Add saving to csv
        print(f'Finished Training \nTotal Time {self.total_time}\n Average Time Per Epoch {(self.total_time)/self.epochs}')

    def print_config(self):
        for opt in self.settings:
            print(f'{opt = }')

    def train_epoch(self,epoch,trainloader):
        self.model.train()
        running_loss = 0
        #Record Time
        time_start = time.perf_counter()
        
        for data in (iter(trainloader)):
            images, labels = data[0].to(self.device),data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if self.binarise:
                for p in (self.model.parameters()):
                    if hasattr(p,'org'):
                        p.data.copy_(p.org)
                self.optimizer.step()
                for p in (self.model.parameters()):
                    if hasattr(p,'org'):
                        p.org.copy_(p.data.clamp_(-1,1))
            else:
                self.optimizer.step()
            running_loss += loss.item()
        self.scheduler.step()
        
        self.total_time +=time.perf_counter()-  time_start
        
        print(f'epoch: {epoch+1} average loss: {running_loss/len(trainloader)} Epoch Time {self.total_time:.1f}')
        
        
        if self.saveNet:
            if running_loss < self.best_loss:
                self.best_loss = running_loss
                self.save('best_loss',epoch,loss)
            self.save('latest',epoch,loss)
        return running_loss

    def valid(self,validloader):
        return self.test(validloader,valid = True)
    def test(self,testloader,valid = False):
        # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        #                                     shuffle=False, num_workers=0)
        #For Batch Norm and Dropoff Training
        self.model.eval()

        #Keep Tally of class accuracy. Only Useful for validation data (Not performed against test data)
        correct = 0
        total = 0
        correct_class = {classname: 0 for classname in self.classes}
        total_class = {classname: 0 for classname in self.classes}
        
        with torch.no_grad():
            for data in (testloader):
                images, labels = data[0].to(self.device),data[1].to(self.device)
                outputs = self.model(images)
                _,predict = torch.max(outputs.data,-1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
                if valid:
                    for label, prediction in zip(labels, predict):
                        if label == prediction:
                            correct_class[self.classes[label]] += 1
                        total_class[self.classes[label]] += 1
                    
        # print accuracy for each class only over validation test
        if valid:
            for classname, correct_count in correct_class.items():
                accuracy = 100 * float(correct_count) / total_class[classname]
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        #Overall Accuracy
        acc = correct/total
        print(f'Overall Accuracy : {correct/total*100:.1f}%')

        if acc > self.best_model_accuracy:
            self.best_model_accuracy = acc
            self.save_Best('best_acc')
        return correct/total
    
    def save(self,filename,epoch,loss):
        PATH = os.path.join( self.cwd,f'{self.name}_{filename}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, PATH)
        print(f'{PATH} saved!')

    def save_Best(self,filename):
        PATH = os.path.join( self.cwd,f'{self.name}_{filename}.pth')
        torch.save(self.model.state_dict(), PATH)
        print(f'{filename}.pth saved!')