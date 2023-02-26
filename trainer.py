from tracemalloc import start
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
from datetime import datetime

import time
import torch.optim as optim

from NN_Thesis.models.binarized_modules import BinarizeConv2d, BinarizeLinear
import wandb


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
    def __init__(self,model:nn.Module,model_name:str,project_name:str,classes = None,seed:int = None,epoch_chkpts:list = None,binarise:bool = False):
        

        self.project_name = project_name

        self.name = model_name        
        self.run_name = f'{self.name}'
        self.start_wall_time = datetime.today().strftime("%Y-%m-%d %H-%M-%S")

        self.model = model
        self.element_info = self.get_param_info()
        self.wandbTableData = [[dtype,num] for dtype,num in self.element_info.items()]
        
        self.binarise = binarise

        #Logging Params
        self.tags = []
        self.total_time = 0
        self.best_model_accuracy = 0
        self.best_loss = float('inf')
        self.epoch_losses = []
        self.epoch_time =0
        self.best_acc = 0


        #Default Training Parameters
        self.lr = 0.1
        self.batch_size = 32
        self.epochs = 200
        self.start_epoch = 0
        self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss() 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.T_max = self.epochs//10

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.T_max)
        
        self.seed = seed
        self.model.to(device = self.device)
        #A dictionary where key is the epoch and value are the settings to apply
        
        self.epoch_chkpts = epoch_chkpts
        if self.epoch_chkpts is None:
            self.epoch_chkpts = []

        #Have a file to store run results
        self.cwd = os.path.join(os.getcwd(),'SavedModels',self.project_name,f'{self.run_name}_{self.start_wall_time}')
        os.makedirs(self.cwd)
        #Datasets
        self.classes = classes
        
        #Flags for saving
        self.saveNet = True
        self.saveData = True

    def set_scheduler(self,scheduler = None,**kwargs):
        if scheduler is None:
            self.T_max = self.epochs//10
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.T_max)
            print('Using Default Cosine Annealing:')
        else:
            self.scheduler = scheduler(self.optimizer,**kwargs)
            print(f'Scheduler Set {self.scheduler.state_dict()}')
    
    def set_optimizer(self,optimizer = None,**kwargs):
        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            print('Using Default (SGD) Optimizer')
        else:
            self.optimizer = optimizer(self.model.parameters(),**kwargs) 
            print(f'Optimizer Set to New: {self.optimizer.state_dict()}')
    
    def train(self,trainloader,testloader,**wandb_kwargs):
        # self.set_scheduler()
        self.batch_size = trainloader.batch_size
        config = self.hyperparameters()

        wandb.init(project= self.project_name,config=config,tags = self.tags,**wandb_kwargs)
        wandb.watch(self.model,log_freq = 250)
        wandb.run.name = self.run_name
        table = wandb.Table(data = self.wandbTableData,columns= ['Param Dtype','Number of Elements'])
        
        print('start')
        self.print_config()

        if self.seed is not None:
            torch.manual_seed(self.seed)
        #Store Values as a tuple (loss,accuracy)
        self.model.to(device = self.device)
        #Get Accuracy and loss at epoch 0
        print('Initial accuracy:')
        self.epoch_time = time.perf_counter()
        (train_loss,_),(test_loss,test_acc)= self.test(trainloader),self.test(testloader) 
        self.epoch_time = time.perf_counter() - self.epoch_time


        
        #Log The Data to WandB
        self.epoch_losses.append( (train_loss,test_loss,test_acc))
        wandb.log(self.metrics(self.start_epoch,train_loss,test_loss,test_acc))

        
        #Add chkpt saving

        #Training Loop
        self.total_time = 0
        for epoch in range(self.start_epoch+1,self.epochs+1):

            #Time Epoch Time
            self.epoch_time = time.perf_counter()

            #Train and Test Results
            train_loss,(test_loss,test_acc)=  (self.train_epoch(epoch,trainloader),self.test(testloader))
            self.epoch_losses.append( (train_loss,test_loss,test_acc))

            self.epoch_time = time.perf_counter() - self.epoch_time
            self.total_time += self.epoch_time
            print(f'Epoch Time (Training + Test) = {self.epoch_time:.2f} seconds')

            #Save at checkpoint
            if epoch in self.epoch_chkpts:
                self.save(f'chkpt_{epoch}',epoch,train_loss)
            self.total_time += self.epoch_time
            
            wandb.log(self.metrics(epoch,train_loss,test_loss,test_acc))
            
        if self.saveData:
            self.to_csv(f'{self.name}')
        print(f'Finished Training: \nTotal Time {self.total_time/3600:2f} hours\n Average Time Per Epoch {(self.total_time)/self.epochs:.2f} seconds')
        wandb.run.summary['Total Time (hours)' ]= self.total_time/3600
        _,__,best_acc = list(zip(*self.epoch_losses))
        wandb.run.summary['Best Accuracy'] =  max(best_acc)
        wandb.log({'Element Info' : table}) 
        
    def hyperparameters(self):
        hyperparameters = {
        'Run Start': self.start_wall_time,
        'start_epoch': self.start_epoch,
        'initial_lr': self.lr,
        'batch_size': self.batch_size,
        'epochs': self.epochs,
        'epoch_chkpts': self.epoch_chkpts,
        'device': self.device ,
        'criterion': self.criterion, 
        'optimizer': self.optimizer, 
        'seed': self.seed,
        'model_architecture': type(self.model),
        'binerised_training': self.binarise,
        'Number of Elements': sum([p.numel() for p in self.model.parameters()])
        }

        return hyperparameters

    def metrics(self,epoch,train_loss,test_loss,test_acc):
        if test_acc > self.best_acc:
            self.best_acc = test_acc
        return {
            'epoch': epoch,
            'training_loss':train_loss,
            'test_loss':test_loss,
            'test_accuracy':test_acc,
            'lr':self.scheduler.get_last_lr()[0],
            'epoch time (s)': self.epoch_time,
            'Current Best Acc': self.best_acc
            }

    def get_param_info(self):
        
        element_info = dict()
        for layer in self.model.parameters():
            if str(layer.dtype) not in element_info.keys():
                element_info[str(layer.dtype)] = layer.numel()
            else:
                element_info[str(layer.dtype)] += layer.numel()
        return element_info

    def print_config(self):
        print(f'Project Name: {self.project_name}, Run Name {self.run_name} \n\n')
        for key,value in self.hyperparameters().items():
            print(f'{key} : {value}')

    def train_epoch(self,epoch,trainloader):
        self.model_train()
        running_loss = 0
        #Record Time

        for data in (iter(trainloader)):
            images, labels = data[0].to(self.device),data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            #Change the weight back to non binarised and then clamp weight between 1 and -1
            if self.binarise:
                for p in (self.model.parameters()):
                    if hasattr(p,'org') and p.requires_grad:
                        p.data.copy_(p.org)
                self.optimizer.step()
                for p in (self.model.parameters()):
                    if hasattr(p,'org') and p.requires_grad:
                        p.org.copy_(p.data.clamp_(-1,1))
            else:
                self.optimizer.step()
            running_loss += loss.item()
        self.scheduler.step()
        
        
        
        print(f'epoch: {epoch} average loss: {running_loss/len(trainloader):.3f}')

        if self.saveNet:
            if running_loss < self.best_loss:
                self.best_loss = running_loss
                self.save('best_loss',epoch,loss)
            self.save('latest',epoch,loss)
        return running_loss/len(trainloader)



    def valid(self,validloader):
        return self.test(validloader,valid = True)
    def test(self,testloader,valid = False):
        self.model_eval()
        #Keep Tally of class accuracy. Only Useful for validation data (Not performed against test data)
        correct = 0
        total = 0
        correct_class = {classname: 0 for classname in self.classes}
        total_class = {classname: 0 for classname in self.classes}
        

        test_loss = 0
        with torch.no_grad():
            for data in (testloader):
                images, labels = data[0].to(self.device),data[1].to(self.device)
                outputs = self.model(images)
                _,predict = torch.max(outputs.data,-1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
                loss = self.criterion(outputs, labels)
                if valid:
                    for label, prediction in zip(labels, predict):
                        if label == prediction:
                            correct_class[self.classes[label]] += 1
                        total_class[self.classes[label]] += 1
                test_loss += loss.item()

        # print accuracy for each class only over validation test
        if valid:
            for classname, correct_count in correct_class.items():
                accuracy = 100 * float(correct_count) / total_class[classname]
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        #Overall Accuracy
        acc = correct/total
        test_loss = test_loss/len(testloader)
        print(f'Test Accuracy : {correct/total*100:.1f}%, Test Loss: {test_loss}')

        if acc > self.best_model_accuracy:
            self.best_model_accuracy = acc
            self.save_Best('best_acc')
        return test_loss, acc

    def model_train(self):
        #Simple wrapper that can be modified if Trainer Class is inherited
        #This allows easy modification for adapter training/Finetuning
        self.model.train()
    def model_eval(self):
        #Simple wrapper that can be modified if Trainer Class is inherited
        #This allows easy modification for adapter training/Finetuning
        self.model.eval()

    def save(self,filename,epoch,loss):
        #To save network and current optimiser state
        PATH = os.path.join( self.cwd,f'{self.name}_{filename}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict':self.scheduler.state_dict(),
            'loss': loss,
            }, PATH)
        # print(f'{PATH} saved!')
    
    def save_as_BNN(self,filename):
        for c in self.model.named_children():
            if isinstance(c[1],BinarizeConv2d,BinarizeLinear):
                c[1].weight.data.sign()
                if hasattr(c[1],'bias'):
                    c[1].bias.data.sign()

        self.save_Best(filename)
        print(f'saved model to {filename}.pth as BNN')
    def save_Best(self,filename):
        #Function to only save Neural network

        PATH = os.path.join( self.cwd,f'{self.name}_{filename}.pth')
        torch.save(self.model.state_dict(), PATH)
        print(f'{filename}.pth saved!')

    def load(self,filename,load_model = False,map_location = None):
        chkpt = torch.load(filename,map_location)
        #I should put this in a loop but for now it is ok
        if load_model:
            pass
            self.model.load_state_dict(chkpt['model_state_dict'])

        print(chkpt.keys())
        self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(chkpt['lr_scheduler_state_dict'])
        self.start_epoch = chkpt['epoch']
        #Display Status
        for key in chkpt.keys():
            if not(load_model and key == 'model_state_dict'):
                continue
            print(f'Successfully loaded {key}\ ! ')
            
    def to_csv(self,name= None):
        if name is None:
            name = self.name + '_data'

        PATH = os.path.join(self.cwd,f'{name}.csv')
        with open(PATH,'w') as f:
            f.write(f'epochs,train_loss,test_loss,test_accuracy\n')    
            for i,data in enumerate(self.epoch_losses):
                d = [str(x) for x in data]
                line = f'{i},{",".join(d)}\n'
                f.write(line)
        print(f'Data Saved to {name}.csv')


class adapter_Trainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.set_optimizer()
    def set_optimizer(self):
        #Only optimise trainable weights
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    def model_train(self):
        #Only turn train on for adapter layers e.g dropout, bn
        for _,layer in self.model.adapter_dict.items():
            layer.train()
        self.model.bn3.train()
        self.model.fc.train()