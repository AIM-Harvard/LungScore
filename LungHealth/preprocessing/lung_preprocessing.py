import os
import SimpleITK as sitk
#from lungmask import mask
import numpy as np
import cv2 
import torch
import monai
from .resample_nrrd import resample_and_resize

# normalize data
def NormalizeData(data):                
    return (data - (-1024)) / ((3071) - (-1024)) 

# unnormalize data (get back to original values)
def unNormalizeData(data):
    return (data*((3071) - (-1024))) + (-1024)

# crop image in x and y, but not in z
def crop_img(img,cropx,cropy):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)   
    #startz = z//2-(cropz//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]

# extract lung given nrrd scan and corresponding lung mask
def lung_extraction(lungmask, nrrd):
        """
        Extract lung given nrrd scan and corresponding lung mask.

        Args:
            lungmask: segmented lung from lungmask model
            nrrd: nrrd scan image (SimpleITK Image object) 
        Returns:
            extracted_lung (numpy array): extracted lung based on the lung health development pipeline
        """       
        
        begin_depth = []
        end_depth = []

        try: 
            # get image array from nrrd and normalize it
            scan = sitk.GetArrayFromImage(nrrd)   
            scan = NormalizeData(scan)    

            # threshold lungmask to binary
            ret, thres = cv2.threshold(lungmask, 0, 1, cv2.THRESH_BINARY) 

            # iterate through slices to find the largest lung area
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
                print('No Lungs to segment: ', scan)
                raise Exception('Exception')
                return
            
            if len(end_depth) == 0:
                end_depth.append(scan.shape[0])

            # extract the whole lung - cropped to its border
            cropped_lung = thres[begin_depth[-1]+1:end_depth[0]-1, yy:yy+hh, xx:xx+ww]
            cropped_volume = scan[begin_depth[-1]+1:end_depth[0]-1, yy:yy+hh, xx:xx+ww]  
           
            only_lung_volume = cropped_volume * cropped_lung
            d, h, w = only_lung_volume.shape[0], only_lung_volume.shape[1], only_lung_volume.shape[2] 

            # pad / crop to lung health development selection size (400, 280, 90) - W, H, D
            padding = monai.transforms.SpatialPad(spatial_size=(480, 512), mode = 'constant', value = 0)
            only_lung_volume_padded = padding(torch.from_numpy(only_lung_volume))  
  
            cropped = crop_img(only_lung_volume_padded, 400, 280)    

            padding_depth = monai.transforms.SpatialPad(spatial_size=(110, 280, 400), method = 'end', mode = 'constant', value = 0)
            cropped = padding_depth(cropped.unsqueeze(0)).squeeze(0) 
            cropped = cropped[:90,:,:]   
           
            # Unnormalize the data to get back to original HU values
            extracted_lung = unNormalizeData(cropped)

            return extracted_lung 

        except Exception as e:
            print('no file/folder or error in loading: ', nrrd) 

