"""
  Deep-learning biomarker for Lung Health Lung segmentation
"""
import yaml
import argparse

import os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
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

base_conf_file_path = 'config/'
conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']

parser = argparse.ArgumentParser(description = 'DCM to NRRD pipeline')

parser.add_argument('--conf',
                    required = False,
                    help = 'Specify the YAML configuration file containing the preprocessing details. ' \
                            + 'Defaults to "lung_segmentation_pipeline.yaml"',
                    choices = conf_file_list,
                    default = "lung_segmentation_pipeline.yaml",
                   )


args = parser.parse_args()

conf_file_path = os.path.join(base_conf_file_path, args.conf)

with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

# input-output Paths
NRRD_folder_path = yaml_conf["io"]["NRRD_folder_path"]
lung_segmentation_folder_path = yaml_conf["io"]["lung_segmentation_folder_path"]

CUDA_VISIBLE_DEVICES =  yaml_conf["preprocessing"]["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

#############################

def seg_lung(NRRD_folder_path):
   
    for scan_id in os.listdir(NRRD_folder_path):
        begin_depth = []
        end_depth = []
        try:
            scan_image = sitk.ReadImage(scan_id)         

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
                print('No Lungs to segment: ', scan_id)
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
           
            cropped = unNormalizeData(cropped)

            np.save(lung_segmentation_folder_path+scan_id, cropped)   

        except Exception as e:
            print('no file/folder or error in loading: ', scan_id) 






