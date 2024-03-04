import torch
from torch.utils.data.dataset import Dataset  
import numpy as np
import os


class Train_set(Dataset):

    def __init__(self, image_path, labels_file):
        
        self.image_path = image_path   

        self.labels_file = labels_file

    def __getitem__(self, i):

        labels = np.load(self.labels_file)
        
        labels = labels.reshape(-1,1)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        label = labels[i,0]   
                     
        img = np.load(os.listdir(os.chdir(self.image_path))[i])
              
        img = torch.tensor(img, dtype = torch.float32)
        
        return img, label

    def __len__(self): 

        return len(os.listdir(os.chdir(self.image_path)))
    
    
class Tune_set(Dataset):

    def __init__(self, image_path, labels_file):
        
        self.image_path = image_path   
        
        self.labels_file = labels_file
         
    def __getitem__(self, i):

        labels = np.load(self.labels_file)

        labels = labels.reshape(-1,1)
        labels = torch.tensor(labels, dtype=torch.int64) 
        
        label = labels[i,0]   
                     
        img = np.load(os.listdir(os.chdir(self.image_path))[i])
       
        img = torch.tensor(img, dtype = torch.float32)
                 
        return img, label

    def __len__(self):

        return len(os.listdir(os.chdir(self.image_path)))


