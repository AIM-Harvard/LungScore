"""
  Deep-learning biomarker for Lung Health inference Mhub pipeline from NRRD to lung health score
"""

""""
Here is a demo!

in terminal run:
  python3 inference.py NRRD_folder_location , ex: /mnt/data/nrrds/img1.nrrd
  output: AI Lung Health Score  / score from 0 to 1 , the higher the score, the more the lung damage
"""

import os
import yaml
import argparse
import matplotlib
import numpy as np
import pandas as pd
import torch  
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# from dataset import Test_set

#import lung segmentation, extraction and preprocessing
from lungage.utils_mhub import seg_lung

#import AI_Lung_Health_Model
from lungage.utils_mhub import CNNModel   #### model architecture
from lungage.utils_mhub import AI_lung_health_model
## -----------------------------

import sys
argument = sys.argv[1]  #path to nrrd scan , ex: /mnt/data/nrrd/img1.nrrd

## ----------------------------------------

def inference(argument, model):
    
    #Lung segmentation, extraction and preprocessing
    extracted_lung = seg_lung(argument)

    #Predicting AI Lung health score
    score = AI_lung_health_model(extracted_lung, model)

    return score


