
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pickle
from torchvision import transforms
def unpickle(file):
    with open(file,'rb') as f:
        return pickle.load(f)

import random
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset, DataLoader


def split_CIFAR100(base = 80,train = True,seed = 42,shift_idx = False,transform = None):
    # Set random seed for reproducibility
    random.seed(seed)

    # Define the transform to be applied to the dataset
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # Fetch the Cifar100 dataset
    cifar100 = CIFAR100(root=r'data\cifar-100', train=train, transform=transform,download=True)
    
    # Get the list of all unique class labels in the dataset


    class_labels = list(set(cifar100.targets))
    classes = cifar100.classes.copy()
    # Shuffle the class labels randomly
    random.shuffle(class_labels)
    random.seed(seed)
    random.shuffle(classes)

    # Split the class labels into two subsets of sizes n and 100-n, respectively
    class_labels_1 = (classes[:base],class_labels[:base])
    class_labels_2 = (classes[base:],class_labels[base:])

    # Filter the dataset to keep only the samples corresponding to the two subsets of class labels
    subset_1_indices = [i for i, target in enumerate(cifar100.targets) if target in class_labels_1[-1]]
    subset_2_indices = [i for i, target in enumerate(cifar100.targets) if target in class_labels_2[-1]]
    subset_1 = Subset(cifar100, subset_1_indices)
    subset_2 = Subset(cifar100, subset_2_indices)


    if shift_idx:
        subset_2 = [(img,label-base) for img,label in subset_2]

    # Create data loaders for the two subsets
    

    # Print the sizes of the two subsets
    print("Subset 1 size:", len(subset_1))
    print("Subset 2 size:", len(subset_2))
    
    for cl in [class_labels_1,class_labels_2]:
        for label,idx in zip(*cl):
            assert label == cifar100.classes[idx]
    print('All classes match up')

    out = {
        'Base': (subset_1,list( zip(*class_labels_1))),
        'Finetune': (subset_2,list(zip(*class_labels_2)))
    }
    return out


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




def apply_transforms(dataset,train= True):
    def normalize_channels(data):
    #We have a nxCxWxH array
        d = data
        d = torch.flatten(data,2,-1).to(dtype = torch.float32)/255
        mean= torch.mean(d,dim = [0,2])
        std = torch.std(d,dim = [0,2])
        return mean,std

    mean,std = normalize_channels(dataset.data)
    if train:
        transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
        transforms.Normalize(mean, std),
        ])

    dataset.transform = transform
    return dataset




def get_split_dataset(train_path,test_path,dataset_type = cifar_n_dataset,train_trainsform = None,test_trainsform = None):

    train_data = dataset_type(train_path)
    test_data = dataset_type(test_path)
    # print(train_data.data.shape)

    def normalize_channels(data):
        #We have a nxCxWxH array
        d = data
        d = torch.flatten(data,2,-1).to(dtype = torch.float32)/255
        mean= torch.mean(d,dim = [0,2])
        std = torch.std(d,dim = [0,2])
        return mean,std

    mean,std = normalize_channels(train_data.data)
    if train_trainsform is None:
        train_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std),
        ])

    if test_trainsform is None:
        test_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_data.transform = train_transform
    test_data.transform = test_transform
    classes = tuple(train_data.label_names)
    classes
    return train_data,test_data,classes


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
