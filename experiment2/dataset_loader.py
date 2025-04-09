import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
import torchvision.datasets as datasets
# pip install --upgrade cryptography
os.environ['http_proxy'] = 'http://172.17.0.2:7532'
os.environ['https_proxy'] = 'http://172.17.0.2:7532'

torch.manual_seed(3407)

def load_dataset(train_dataset_path, val_dataset_path, batch_size=32, is_train_shuffle=True, is_val_shuffle=False):
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010]) 
    ])
    
    train_dataset = datasets.CIFAR10(root=train_dataset_path, train=True, 
                                    transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(root=val_dataset_path, train=False, 
                                  transform=val_transform, download=True)    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_train_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=is_val_shuffle)

    return train_loader, val_loader

if __name__ == "__main__":
    train_dataset_path = "data/train"
    val_dataset_path = "data/val"
    batch_size = 32
    is_train_shuffle = True
    is_val_shuffle = False
    train_loader, val_loader = load_dataset(train_dataset_path, val_dataset_path, batch_size, is_train_shuffle, is_val_shuffle)

    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels)
        break