import os
import pickle
from pathlib import Path
import torch.nn.functional as F
import requests

import torch
import torch.nn as nn
import wget
from .model import lungage

def lungage_load(model = lungage, eval_mode=True, device = "cuda" if torch.cuda.is_available() else "cpu"):

    """
    Load Lung Health Model 
    Args:
        model (str): lungage  - model name to call - 
        eval_mode: True to run model in evaluation mode 
    Returns:
        AI_Lung_health_score --> A score between 0 to 1 - 1: most healthy lung, while 0 is most damaged lung
    """
    # call the model to device
    model = lungage().to(device)

    
    # download model weights
    weights_url = "https://zenodo.org/records/14065852/files/AI_Lung_Health_Model.pth"
    current_path = Path(os.getcwd())

    if not (current_path / "model_weights.pth").exists():
        wget.download(weights_url, out=os.path.join(os.getcwd(), 'model_weights.pth'))

    # Load the pretrained weights
    model = nn.DataParallel(model, device_ids = [0, 2, 3])

    model.load_state_dict(torch.load(current_path / "model_weights.pth", map_location=device))
    
    if eval_mode:
        model.eval()

    return model

def lungage_predict(model, extracted_lung, device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predict AI lung health score given the segmented lung
    Args:
        model: lung health model to 
        extracted lung: segmented lung from NRRD Scan 
    Returns:
        AI_Lung_health_score --> A score between 0 to 1 - 1: most healthy lung, while 0 is most damaged lung
    """
    ai_lungage_score = F.softmax(model(extracted_lung.to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)).cpu().detach(), dim=1).numpy()[:, 0] # higher score means healthier lung therefore better outcome 

    return ai_lungage_score

