
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pickle
def unpickle(file):
    with open(file,'rb') as f:
        return pickle.load(f)


class cifar_n_dataset(Dataset):
    def __init__(self,file_dir,transform = None):
        self.transform = transform
        self.cifar_dir = file_dir

        for key,value in unpickle(file_dir).items():
            setattr(self,key,value)    
        self.data = self.data.reshape((self.data.shape[0],3,32,32))
        self.data = torch.from_numpy(self.data).to(dtype=torch.float32)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img,label



class cifar_100_split(Dataset):
    def __init__(self,root,train = True,transform = None):
        self.transform = transform

        f = 'train' if train else 'test'
        self.data = unpickle(os.path.join(root,f))
        self.data,self.label =  list(zip(*[ (torch.from_numpy(img).to(dtype = torch.float32),label) for img,label in self.data]))
        self.data = torch.stack(self.data)/255
        self.classes = unpickle(os.path.join(root,'classes'))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img,label = self.data[idx],self.label[idx]
        if self.transform:
            img = self.transform(img)
        return img,label





from torch.utils.data import DataLoader

if __name__ == '__main__':
    PATH = 'data/cifar_5/cifar_04/train/data'
    train_dataset = cifar_n_dataset(PATH)
    train_dataloader = DataLoader(train_dataset,shuffle = True,batch_size= 2)

    x,_ = train_dataset[0]
    print(x.shape)
    for i in range(4):
        img, label = next(iter(train_dataloader))
        img = img[0].squeeze()
        # print(img.shape)
        img = torch.moveaxis(img,0,-1)
        classes = train_dataset.label_names
        print(classes[label[0]])
        plt.imshow(img.squeeze())
        plt.show()
