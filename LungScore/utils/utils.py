"""
  Deep-learning biomarker for Lung Health utils - calculate lung volume - check corrupted/outofrange values - clipping pixel values
"""
import numpy as np
import torch


def lung_vol_pix(seg_lung):

    """
    count number of pixels as a proxy for lung volume 

    Args:
        seg_lung (numpy_array): numpy array of segmented lung
    Returns:
        lung volume (float): represented by counting the pixels values that are not representing air (not equal to -1024)
    """

    lung_volume = np.sum(seg_lung != -1024)
    
    return lung_volume


def scans_w_outofrange_values(seg_lung):

    """
    check out of range / corrupted pixel values in segmented lung > 3071 or < -1024
    Args:
        seg_lung (numpy_array): numpy array of segmented lung
    Returns: 
        Print statment with corrupted max and min values or no corrupted values 
        
        
    """

    if (seg_lung.max() > (3071) or (seg_lung.min()<(-1024))):

        print("Corrupted or out of range values, max value = ", seg_lung.max())
        print("Corrupted or out of range values, min value = ", seg_lung.min())

    else:
        print("No corrupted / out of range values")
    
    return None


def clipping_values(seg_lung_tensor):
    """
    clipping pixel values to 3071 and -1024

    Args:
        seg_lung_tensor (torch tensor): torch tensor of segmented lung
    Returns:
        seg_lung_tensor_clipped (tensor): segmented lung tensor with values clipped to 3071 and -1024
    """

    seg_lung_tensor_clipped = torch.clamp(seg_lung_tensor, min=-1024, max=3071)
    
    return seg_lung_tensor_clipped



 