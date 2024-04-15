"""
  Deep-learning biomarker for Lung Health - Mhub utils
"""
import yaml
import argparse

import os
import SimpleITK as sitk
import shutil
from lungmask import mask
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
import pandas as pd
import pickle
import cv2  
import seaborn as sns
import torch
import monai

import torch
import os
import torch.nn as nn
import torch.nn.functional as F


def NormalizeData(data):                
    return (data - (-1024)) / ((3071) - (-1024)) 

def unNormalizeData(data):
    return (data*((3071) - (-1024))) + (-1024)

def crop_img(img,cropx,cropy):

    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)   
    #startz = z//2-(cropz//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]

## Lung segmentation, preprocessing and extraction

def seg_lung(scan_folder):
   
        begin_depth = []
        end_depth = []

        try:
            scan_image = sitk.ReadImage(scan_folder)         

            scan = sitk.GetArrayFromImage(scan_image)   
        
            scan = NormalizeData(scan)    
            
            lung_mask = mask.apply(scan_image)

            ret, thres = cv2.threshold(lung_mask, 0, 1, cv2.THRESH_BINARY) 

            max_area_slice = 0 
            for slc_no in range(scan.shape[0]):
                contours, hierarchy = cv2.findContours(thres[slc_no],  
                            cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
                mx = (0,0,0,0)
                mx_area = 0
                areas = []
                dims = [] 
                for cont in contours:
                    x,y,w,h = cv2.boundingRect(cont)
                    area = w*h
                    areas.append(area)
                    dims.append([x,y,w,h])
                    if area > mx_area:
                        mx = x,y,w,h
                        mx_area = area
                    
                x,y,w,h = mx
    
                if (len(areas) == 0) and (max_area_slice == 0): 
                    begin_depth.append(slc_no)
                
                elif (len(areas) == 0) and (max_area_slice > 0):
                    end_depth.append(slc_no)
                
                elif len(areas) == 1:
                    x, y, w, h = x, y, w, h
                    
                else: 
                    smallest_area = 0 
                    idx = 0
                    idx_smallest = 0
                    for area in areas: 
                        if (area > smallest_area) and (area < mx_area):
                            smallest_area = area
                            idx_smallest = idx
                        idx += 1
                    
                    idx_largest = 0
                    mx_area = mx_area
                    idx_ = 0
                    for area in areas:
                        if area == mx_area: 
                            idx_largest = idx_
                        idx_ += 1
                        
                    if dims[idx_smallest][1] > dims[idx_largest][1]:
                        y = dims[idx_largest][1]
                    else:  
                        y = dims[idx_smallest][1] 
                    
                    if dims[idx_smallest][0] > dims[idx_largest][0]:
                        x = dims[idx_largest][0]
                    else:  
                        x = dims[idx_smallest][0] 
                        
                    if dims[idx_smallest][0] + dims[idx_smallest][2] > dims[idx_largest][0] + dims[idx_largest][2]:
                        w = dims[idx_smallest][0] + dims[idx_smallest][2] - x
                    else:  
                        w =  dims[idx_largest][0] + dims[idx_largest][2] - x
            
                    if dims[idx_smallest][1] + dims[idx_smallest][3] > dims[idx_largest][1] + dims[idx_largest][3]:
                        h = dims[idx_smallest][1] + dims[idx_smallest][3] - y
                    else:  
                        h =  dims[idx_largest][1] + dims[idx_largest][3] - y
                
                    area_slice = w*h
                    
                    if area_slice > max_area_slice: 
                        max_area_slice = area_slice 
                        xx, yy, ww, hh = x, y, w, h
                    
                        largest_slice = slc_no
            
            if len(begin_depth) == 0:
                print('No Lungs to segment: ', scan_folder)
                raise Exception('Exception')
                return
            
            if len(end_depth) == 0:
                end_depth.append(scan.shape[0])

            cropped_lung = thres[begin_depth[-1]+1:end_depth[0]-1, yy:yy+hh, xx:xx+ww]
            cropped_volume = scan[begin_depth[-1]+1:end_depth[0]-1, yy:yy+hh, xx:xx+ww]  
           
            only_lung_volume = cropped_volume * cropped_lung
            d, h, w = only_lung_volume.shape[0], only_lung_volume.shape[1], only_lung_volume.shape[2] 

            padding = monai.transforms.SpatialPad(spatial_size=(480, 512), mode = 'constant', value = 0)
            only_lung_volume_padded = padding(torch.from_numpy(only_lung_volume))  
  
            cropped = crop_img(only_lung_volume_padded, 400, 280)    

            padding_depth = monai.transforms.SpatialPad(spatial_size=(110, 280, 400), method = 'end', mode = 'constant', value = 0)
            cropped = padding_depth(cropped.unsqueeze(0)).squeeze(0) 
            cropped = cropped[:90,:,:]   
           
            extracted_lung = unNormalizeData(cropped)

            return extracted_lung

        except Exception as e:
            print('no file/folder or error in loading: ', scan_folder) 


###################################

# AI_LUNG_HEALTH Model
class CNNModel(nn.Module): 
    def __init__(self): 
        super(CNNModel, self).__init__()  
 
        self.conv_layer1 = self._conv_layer_set1(1, 16) 
        self.conv_layer2 = self._conv_layer_set234(16, 32) 
        self.conv_layer3 = self._conv_layer_set234(32, 64) 
        self.conv_layer4 = self._conv_layer_set234(64, 128)
        self.conv_layer5 = self._conv_layer_set5(128, 256) 

        self.fc1 = nn.Linear(15360, 1024)    #15360 
        self.fc2 = nn.Linear(1024, 128) 
        self.fc3 = nn.Linear(128, 2)    
 
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.4)            
          
    def _conv_layer_set1(self, in_c, out_c):  
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(7, 7, 7), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2,2,2)), 
        nn.BatchNorm3d(out_c),
        nn.Dropout(p=0.2)
        ) 
        return conv_layer  
 
    def _conv_layer_set234(self, in_c, out_c): 
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),  
        nn.LeakyReLU(),
        nn.MaxPool3d((2,2,2)),
        nn.BatchNorm3d(out_c),
        nn.Dropout(p=0.2) 
        ) 
        return conv_layer 

    def _conv_layer_set5(self, in_c, out_c): 
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),  
        nn.LeakyReLU(),
        nn.MaxPool3d((2,2,2), padding=(1,0,0)),
        nn.BatchNorm3d(out_c),
        nn.Dropout(p=0.2) 
        ) 
        return conv_layer 


    def NormalizeData(self, data):
         return (data - (-1024)) / ((1566) - (-1024))     # original lung health normalization

    def forward(self, x):   

        out = self.NormalizeData(x)  
   
        out = self.conv_layer1(out)     
        out = self.conv_layer2(out) 
        out = self.conv_layer3(out) 
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)  
       
        out = out.view(out.size(0), -1)

        out = self.drop(self.relu(self.fc1(out)))
        out = self.drop(self.relu(self.fc2(out)))
        out = self.fc3(out) 

        return out
    

##############################################

# Predicting AI_lung_health_score
def AI_lung_health(extracted_lung, model, device = torch.device("cuda") ):

    model.eval()
    with torch.no_grad():
        pred = model.to(device)(extracted_lung.to(device).unsqueeze(0).unsqueeze(0))      
        ai_lung_health_score = F.softmax(pred.cpu().detach(), dim=1).numpy()[:, 1] 

    return ai_lung_health_score 


