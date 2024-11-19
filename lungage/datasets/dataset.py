import torch
from torch.utils.data.dataset import Dataset  
import numpy as np
import os
from PIL import Image
import pandas as pd


class Test_set(Dataset):

    def __init__(self, image_path):
        
        self.image_path = image_path   

    def __getitem__(self, i):

        scan = os.listdir(self.image_path)[i]
 
        img = np.load(os.listdir(os.chdir(self.image_path))[i])
       
        img = torch.tensor(img, dtype = torch.float32)
                 
        return img, scan
    
"""
# csv only with image paths
class Test_set(Dataset):
    def __init__(self, path_dir_to_imgs, csv_file):
        
        Args:
            csv_file (string): Path to the CSV file with image paths
            path_dir_to_imgs(string): Directory with all the images
        
        self.data = pd.read_csv(csv_file)  # Load the CSV file
        self.path_dir_to_imgs = path_dir_to_imgs     # Root directory for images

        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve image path and label from the CSV
        img_path = os.path.join(self.path_dir_to_imgs, self.data.iloc[idx, 0])
        scan_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # read the image
        image = np.load(img_path)
        image = torch.tensor(image, dtype = torch.float32)

        return image, scan_name
    
"""

class Train_set(Dataset):
    def __init__(self, path_dir_to_imgs, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with image paths and labels.
            path_dir_to_imgs(string): Directory with all the images.
        """
        self.data = pd.read_csv(csv_file)  # Load the CSV file
        self.path_dir_to_imgs = path_dir_to_imgs     # Root directory for images

        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve image path and label from the CSV
        img_path = os.path.join(self.path_dir_to_imgs, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]
        
        # read the image
        image = np.load(img_path)
        image = torch.tensor(image, dtype = torch.float32)

        # Convert label to tensor if it's not
        label = torch.tensor(label, dtype=torch.int64)

        return image, label
  
class Tune_set(Dataset):
    def __init__(self, path_dir_to_imgs, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with image paths and labels.
            path_dir_to_imgs(string): Directory with all the images.
        """
        self.data = pd.read_csv(csv_file)  # Load the CSV file
        self.path_dir_to_imgs = path_dir_to_imgs     # Root directory for images

        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve image path and label from the CSV
        img_path = os.path.join(self.path_dir_to_imgs, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]
        
        # read the image
        image = np.load(img_path)
        image = torch.tensor(image, dtype = torch.float32)

        # Convert label to tensor if it's not
        label = torch.tensor(label, dtype=torch.int64)

        return image, label