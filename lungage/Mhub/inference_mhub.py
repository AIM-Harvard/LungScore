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

## these steps are made by Mhub
## 1- DICOM TO NRRD 
## 2- loading model weights

#import lung segmentation, extraction and preprocessing
from lungage.Mhub import extract_lung # step 1: extract lung

#import AI_Lung_Health_Model
from lungage.Mhub import lungage   # model architecture
from lungage.Mhub import AI_lung_health # step 2: predict ai_lung_health_score
## -----------------------------

import sys
argument = sys.argv[1]  #path to nrrd scan , ex: /mnt/data/nrrd/img1.nrrd

## ----------------------------------------

def inference(argument, model):
    
    #step1: Lung segmentation, extraction and preprocessing
    extracted_lung = extract_lung(argument)

    #step2: Predicting AI Lung health score
    score = AI_lung_health(extracted_lung, model)

    return score


