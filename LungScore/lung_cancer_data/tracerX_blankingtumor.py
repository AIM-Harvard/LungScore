import os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
import SimpleITK as sitk
import shutil
from lungmask import mask
#from lungmask import LMInferer
#from lungmask.mask import LMInferer, apply, apply_fused
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
import pandas as pd
import pickle
import cv2  
import pandas as pd
import seaborn as sns
import torch
from skimage.transform import rescale, resize, downscale_local_mean
import monai
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    Resize
)
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 


def resample_and_resize(image_path, new_spacing):
    # Load the NRRD image
    image = sitk.ReadImage(image_path)
   
    # print(image.GetSize())
    # print(image.GetSpacing())

    orig_size = image.GetSize()
    orig_spacing = image.GetSpacing()

    new_size = [int(orig_size[0] * orig_spacing[0] / new_spacing[0]),
                int(orig_size[1] * orig_spacing[1] / new_spacing[1]),
                int(orig_size[2] * orig_spacing[2] / new_spacing[2])]

    # Resample the image to the new spacing
    original_spacing = image.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    # resampler.SetSize([int(sz * osp / nsp) for sz, osp, nsp in zip(image.GetSize(), original_spacing, new_spacing)])

    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputPixelType(image.GetPixelID())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    resampled_image = resampler.Execute(image)
    
    resampled_image_size = resampled_image.GetSize()

    # print(resampled_image.GetSize())
    # print(resampled_image.GetSpacing())

    # Resize the resampled image to the new size
    #resized_image = sitk.Resample(resampled_image, new_size, sitk.Transform(), sitk.sitkLinear, resampled_image.GetOrigin(), new_spacing, resampled_image.GetDirection(), 0.0, resampled_image.GetPixelID())
    
    final_size = [512, 512, 0]

    old_size = resampled_image.GetSize()


    new_size_down = [max(0, int((old_size[0] - final_size[0]) / 2)),
                    max(0, int((old_size[1] - final_size[1]) / 2)),
                    0]
    new_size_up = [max(0, old_size[0] - final_size[0] - new_size_down[0]),
                    max(0, old_size[1] - final_size[1] - new_size_down[1]),
                    0]
    
    crop_filter = sitk.CropImageFilter()
    crop_filter.SetUpperBoundaryCropSize(new_size_up)
    crop_filter.SetLowerBoundaryCropSize(new_size_down)
    resized_image = crop_filter.Execute(resampled_image)


    old_size = resized_image.GetSize()

    new_size_down = [max(0, int((final_size[0] - old_size[0]) / 2)),
                    max(0, int((final_size[1] - old_size[1]) / 2)),
                    0]
    new_size_up = [max(0, final_size[0] - old_size[0] - new_size_down[0]),
                    max(0, final_size[1] - old_size[1] - new_size_down[1]),
                    0]

    air = -1024
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetConstant(air)
    pad_filter.SetPadUpperBound(new_size_up)
    pad_filter.SetPadLowerBound(new_size_down)
    resized_image = pad_filter.Execute(resized_image)

    # print(resized_image.GetSize())
    # print(resized_image.GetSpacing())

    
    return resized_image, orig_spacing


def resample_and_resize_tumor(image_path, new_spacing):
    # Load the NRRD image
    image = sitk.ReadImage(image_path)
   
    # print(image.GetSize())
    # print(image.GetSpacing())

    orig_size = image.GetSize()
    orig_spacing = image.GetSpacing()

    new_size = [int(orig_size[0] * orig_spacing[0] / new_spacing[0]),
                int(orig_size[1] * orig_spacing[1] / new_spacing[1]),
                int(orig_size[2] * orig_spacing[2] / new_spacing[2])]

    # Resample the image to the new spacing
    original_spacing = image.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    # resampler.SetSize([int(sz * osp / nsp) for sz, osp, nsp in zip(image.GetSize(), original_spacing, new_spacing)])

    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputPixelType(image.GetPixelID())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    resampled_image = resampler.Execute(image)
    
    resampled_image_size = resampled_image.GetSize()

    # print(resampled_image.GetSize())
    # print(resampled_image.GetSpacing())

    # Resize the resampled image to the new size
    #resized_image = sitk.Resample(resampled_image, new_size, sitk.Transform(), sitk.sitkLinear, resampled_image.GetOrigin(), new_spacing, resampled_image.GetDirection(), 0.0, resampled_image.GetPixelID())
    
    final_size = [512, 512, 0]

    old_size = resampled_image.GetSize()


    new_size_down = [max(0, int((old_size[0] - final_size[0]) / 2)),
                    max(0, int((old_size[1] - final_size[1]) / 2)),
                    0]
    new_size_up = [max(0, old_size[0] - final_size[0] - new_size_down[0]),
                    max(0, old_size[1] - final_size[1] - new_size_down[1]),
                    0]
    
    crop_filter = sitk.CropImageFilter()
    crop_filter.SetUpperBoundaryCropSize(new_size_up)
    crop_filter.SetLowerBoundaryCropSize(new_size_down)
    resized_image = crop_filter.Execute(resampled_image)


    old_size = resized_image.GetSize()

    new_size_down = [max(0, int((final_size[0] - old_size[0]) / 2)),
                    max(0, int((final_size[1] - old_size[1]) / 2)),
                    0]
    new_size_up = [max(0, final_size[0] - old_size[0] - new_size_down[0]),
                    max(0, final_size[1] - old_size[1] - new_size_down[1]),
                    0]

    air = -1024
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetConstant(air)
    pad_filter.SetPadUpperBound(new_size_up)
    pad_filter.SetPadLowerBound(new_size_down)
    resized_image = pad_filter.Execute(resized_image)

    # print(resized_image.GetSize())
    # print(resized_image.GetSpacing())

    
    return resized_image

# Example usage
# image_path = "/mnt/data8/TracerX_baseline_simon_copy_Ahmed/LTX1000_img.nrrd"
# new_spacing = [0.68, 0.68, 2.5]  # New spacing (depth, height, width)
# resized_image = resample_and_resize(image_path, new_spacing)


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

ss = 0
trx_tumor_in_leftlung = []
trx_tumor_in_rightlung = []

chan = []

# trx = np.load('/mnt/data6/DeepPY/ai_lung_damage/survival_analysis/trx_rerun_patids.npy')
# print(trx)

# for vol in trx:
pat_ids = []
cent_of_blank_loc = []


# for vol in os.listdir('/mnt/data6/DeepPY/ai_lung_damage/data/trx_segmented_lungs/'):


#for vol in file_path:
for vol in os.listdir('/mnt/data6/DeepPY/ai_lung_damage/data/trx_segmented_lungs/'):
    print(vol)                     
    print(ss)
    ss+=1    

    volume = vol[:7]
    #volume = vol[-70:]
    print(volume)

    begin_depth = []
    end_depth = []

    not_valid_first  = []
    not_valid = []
   
    try: 
     	if (volume+'_img.nrrd' in os.listdir('/mnt/data8/TracerX_baseline_simon_copy_Ahmed/')):
       #if (ss==ss):
 
           vol_ = sitk.ReadImage('/mnt/data8/TracerX_baseline_simon_copy_Ahmed/'+volume+'_img.nrrd') 
  
           tumor = sitk.ReadImage('/mnt/data8/TracerX_baseline_simon_copy_Ahmed/'+volume+'_msk.nrrd')          
 
##############################
           
           new_spacing = [0.68, 0.68, 2.5]
           image_path = '/mnt/data8/TracerX_baseline_simon_copy_Ahmed/'+volume+'_img.nrrd'
           tumor_path = '/mnt/data8/TracerX_baseline_simon_copy_Ahmed/'+volume+'_msk.nrrd'

           vol_, orig_spacing = resample_and_resize(image_path, new_spacing)   
           tumor = resample_and_resize_tumor(tumor_path, new_spacing) 

        #    nrrd_writer = sitk.ImageFileWriter()
        #    nrrd_file = os.path.join('/mnt/data6/DeepPY/ai_lung_damage/data/trx_tumor_masked/yes.nrrd')
        #    nrrd_writer.SetFileName(nrrd_file)
        #    nrrd_writer.SetUseCompression(True)
        #    nrrd_writer.Execute(vol_)
           

#################
                 
           vol = sitk.GetArrayFromImage(vol_) 
        
           tumor = sitk.GetArrayFromImage(tumor)   
           
           vol = NormalizeData(vol)    

           msk = mask.apply(vol_)

           ret, thres = cv2.threshold(msk, 0, 1, cv2.THRESH_BINARY) 

           max_area_slice = 0 
           not_valid = []
           for slc_no in range(vol.shape[0]):
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
              begin_depth.append(-1)

           if len(end_depth) == 0:
              end_depth.append(vol.shape[0]) 
      
           lung_tumor = tumor[begin_depth[-1]+1:end_depth[0], yy:yy+hh, xx:xx+ww]
           
           if (lung_tumor.max() > 0) and (orig_spacing[2] < 3.5):

                cropped_lung = thres[begin_depth[-1]+1:end_depth[0], yy:yy+hh, xx:xx+ww]
                cropped_volume = vol[begin_depth[-1]+1:end_depth[0], yy:yy+hh, xx:xx+ww]  
        
                only_lung_volume = cropped_volume * cropped_lung

                d, h, w = only_lung_volume.shape[0], only_lung_volume.shape[1], only_lung_volume.shape[2] 
                #print(only_lung_volume)
               ########### 

                mask_slices_to_black_out = (lung_tumor > 0).any(axis=(1, 2))
                num_channels_to_black_out = np.sum(mask_slices_to_black_out)
                chan.append(num_channels_to_black_out)

                #print(mask_slices_to_black_out)
                
                if num_channels_to_black_out == 28 and (np.where(mask_slices_to_black_out)[0][-1] <= 89) and (np.where(mask_slices_to_black_out)[0][0] >= 0):
                    only_lung_volume[mask_slices_to_black_out] = 0
                    channel_indices_to_black_out = np.where(mask_slices_to_black_out)[0]

                    indices = channel_indices_to_black_out
                    for i in range(len(indices) - 1):
                         if indices[i + 1] - indices[i] > 1:  # more than one tumor 
                             not_valid_first = 'NO'
                             break

                 #    print(channel_indices_to_black_out)

                elif (num_channels_to_black_out == 28)  and (np.where(mask_slices_to_black_out)[0][-1] > 89 or np.where(mask_slices_to_black_out)[0][0] < 0):
                   
                    remaining = (28 - num_channels_to_black_out) / 2  

                    rounded_up = math.ceil(remaining)  
                    rounded_down = math.floor(remaining) 

                    channel_indices_to_black_out = np.where(mask_slices_to_black_out)[0]
                    print(channel_indices_to_black_out)
                   
                    indices = channel_indices_to_black_out
                    for i in range(len(indices) - 1):
                         if indices[i + 1] - indices[i] > 1:
                             not_valid_first = 'NO'
                             break

                    start = channel_indices_to_black_out[0]
                    end = channel_indices_to_black_out[-1]
                    start = start - rounded_down
                    end = end + rounded_up
                   
                    print(start)
                    print(end)

                    if start >= 0 and end < lung_tumor.shape[0]:
                       mask_slices_to_black_out[start:end+1] = True  # so total is 28 slices blanked
                       print(np.where(mask_slices_to_black_out)[0])
                    
                    elif start >= 0 and end >= lung_tumor.shape[0]:
                       print(end)
                       print(start)
                       print(lung_tumor.shape[0])
                       start = start - (end+1 - lung_tumor.shape[0])
                       end = lung_tumor.shape[0]
                       mask_slices_to_black_out[start:end] = True
                       print(np.where(mask_slices_to_black_out)[0])

                    elif start < 0 and end < lung_tumor.shape[0]:
                       end = end + (0-start)
                       start = 0                    
                       mask_slices_to_black_out[start:end+1] = True
                       print(np.where(mask_slices_to_black_out)[0])

                   
               #     ############################### make sure these start and end fits the cropping size 0-89
                      
                    if start >= 0 and end < 90:
                       only_lung_volume[mask_slices_to_black_out] = 0
                       print(np.where(mask_slices_to_black_out)[0])
                    
                    elif (start < 0):
                        more_at_back = (0 - start)
                        mask_slices_to_black_out[0:end+more_at_back+1] = True
                        mask_slices_to_black_out[end+more_at_back + 1:] = False

                        only_lung_volume[mask_slices_to_black_out] = 0
                        print(np.where(mask_slices_to_black_out)[0])

                    elif (end >= 90):
                        more_at_front = ((end - 89))
                        if end == lung_tumor.shape[0]:
                           mask_slices_to_black_out[start-more_at_front+1:end+1-more_at_front] = True
                        else:
                           mask_slices_to_black_out[start-more_at_front:end+1-more_at_front] = True

                        mask_slices_to_black_out[:start - more_at_front] = False
                        mask_slices_to_black_out[end + 1 - more_at_front:] = False

                        only_lung_volume[mask_slices_to_black_out] = 0
                        print(np.where(mask_slices_to_black_out)[0])                   
                   
                elif (num_channels_to_black_out < 28):
                   
                    remaining = (28 - num_channels_to_black_out) / 2  

                    rounded_up = math.ceil(remaining)  
                    rounded_down = math.floor(remaining) 

                    channel_indices_to_black_out = np.where(mask_slices_to_black_out)[0]
                    print(channel_indices_to_black_out)
                   
                    indices = channel_indices_to_black_out
                    for i in range(len(indices) - 1):
                         if indices[i + 1] - indices[i] > 1:
                             not_valid_first = 'NO'
                             break

                    start = channel_indices_to_black_out[0]
                    end = channel_indices_to_black_out[-1]
                    start = start - rounded_down
                    end = end + rounded_up
                   
                    print(start)
                    print(end)
                   
               #     ########################### make sure the new start and end still fits the image size 
                 
                    if start >= 0 and end < lung_tumor.shape[0]:
                       mask_slices_to_black_out[start:end+1] = True  # so total is 28 slices blanked
                       print(np.where(mask_slices_to_black_out)[0])
                      
                    elif start >= 0 and end >= lung_tumor.shape[0]:
                       print(end)
                       print(start)
                       print(lung_tumor.shape[0])
                       start = start - (end+1 - lung_tumor.shape[0])
                       end = lung_tumor.shape[0]
                       mask_slices_to_black_out[start:end] = True
                       print(np.where(mask_slices_to_black_out)[0])

                    elif start < 0 and end < lung_tumor.shape[0]:
                       end = end + (0-start)
                       start = 0                    
                       mask_slices_to_black_out[start:end+1] = True
                       print(np.where(mask_slices_to_black_out)[0])

                   
                    ############################### make sure these start and end fits the cropping size 0-89
                      
                    if start >= 0 and end < 90:
                       only_lung_volume[mask_slices_to_black_out] = 0
                       print(np.where(mask_slices_to_black_out)[0])
                    
                    elif (start < 0):
                        more_at_back = (0 - start)
                        mask_slices_to_black_out[0:end+more_at_back+1] = True
                        mask_slices_to_black_out[end+more_at_back + 1:] = False

                        only_lung_volume[mask_slices_to_black_out] = 0
                        print(np.where(mask_slices_to_black_out)[0])

                    elif (end >= 90):
                        more_at_front = ((end - 89))
                        if end == lung_tumor.shape[0]:
                           mask_slices_to_black_out[start-more_at_front+1:end+1-more_at_front] = True
                        else:
                           mask_slices_to_black_out[start-more_at_front:end+1-more_at_front] = True

                        mask_slices_to_black_out[:start - more_at_front] = False
                        mask_slices_to_black_out[end + 1 - more_at_front:] = False

                        only_lung_volume[mask_slices_to_black_out] = 0
                        print(np.where(mask_slices_to_black_out)[0])
    
               
                elif num_channels_to_black_out > 28:
                    not_valid = 'NO'
               
               ############

           padding = monai.transforms.SpatialPad(spatial_size=(480, 512), mode = 'constant', value = 0)
           only_lung_volume_padded = padding(torch.from_numpy(only_lung_volume))  
        
           cropped = crop_img(only_lung_volume_padded, 400, 280)    

           padding_depth = monai.transforms.SpatialPad(spatial_size=(110, 280, 400), method = 'end', mode = 'constant', value = 0)
           cropped = padding_depth(cropped.unsqueeze(0)).squeeze(0) 
           cropped = cropped[:90,:,:]   
           cropped = unNormalizeData(cropped)
           #print(cropped)

           if not_valid == 'NO' or not_valid_first == 'NO':
              print('wont save!')
              pass
          
           else:
              np.save('/mnt/data6/DeepPY/ai_lung_damage/data/trx_wl_tumorsliceremovedmax/'+ volume + 'tumorremovedmaxwl', cropped) 

    except Exception as e:
        print('no file/folder or error in loading the data') 

