import os
import pickle
from pathlib import Path
import torch.nn.functional as F
import requests

import torch
import torch.nn as nn
import wget
from .model import Lungscore

def lungscore_load(model = Lungscore, eval_mode=True, device = "cuda" if torch.cuda.is_available() else "cpu"):

    """
    Load Lung Score Model 
    Args:
        model (str): lungscore  - model name to call - 
        eval_mode: True to run model in evaluation mode 
    Returns:
        model: Lung Score model with loaded pretrained weights
    """
    # call the model to device
    model = Lungscore().to(device)

    # download model weights
    weights_url  = "https://zenodo.org/records/15829020/files/Modelweights_AILH" 
    current_path = Path(os.getcwd())

    if not (current_path / "Modelweights_AILH").exists():
        wget.download(weights_url, out=os.path.join(os.getcwd(), 'Modelweights_AILH'))

    # Load the pretrained weights
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(current_path / "Modelweights_AILH", map_location=device))
    
    # model in eval mode
    if eval_mode:
        model.eval()

    return model

def lungscore_predict(model, extracted_lung, device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predict AI lung score given the segmented lung
    Args:
        model: lung score model loaded in eval mode
        extracted lung: segmented lung from NRRD Scan 
    Returns:
        ai_lung_score (float): lung score between 0 to 1 -- 1: healthiest lung
    """
    
    ai_lung_score = F.softmax(model(extracted_lung.to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)).cpu().detach(), dim=1).numpy()[:, 0] # higher score means healthier lung therefore better outcome 

    # extracted_lung.requires_grad_(True)
    # ai_lung_score = model(extracted_lung)[:,0]  # shape: [1, num_classes]
    
    return ai_lung_score

