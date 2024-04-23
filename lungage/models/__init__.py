import os
import pickle
from pathlib import Path
import torch.nn.functional as F

import torch
import torch.nn as nn
import wget
from .model import lungage

def lungage_load(model = lungage, eval_mode=True, device = "cuda" if torch.cuda.is_available() else "cpu"):

    model = lungage().to(device)

    weights_url = "https://zenodo.org/records/11047105?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxMzg3MzU1OCwiZXhwIjoxODkzNDU1OTk5fQ.eyJpZCI6IjUxMDE5NWIzLTg4OGUtNGViMS05YjU1LWQ5OGRmODc4N2YzOCIsImRhdGEiOnt9LCJyYW5kb20iOiIzNmZmODVjYmVlM2Q2NzFhM2E3Yzc4NGI5NzU0ZmJkNCJ9.f8-n5LbLSV-nOWOpok0fyTrxrWlBBvi_gdICJHStHQSvriWI5BcxYdHHPytpicGNMy1y-rpm07XOw988QuszqQ"
    current_path = Path(os.getcwd())

    if not (current_path / "model_weights.torch").exists():
        wget.download(weights_url, bar=bar_progress)

    # Load the pretrained weights
    model.load_state_dict(torch.load(current_path / "model_weights.torch", map_location=device))
   
    if eval_mode:
        model.eval()

    return model

def lungage_predict(model, extracted_lung, device = "cuda" if torch.cuda.is_available() else "cpu"):

    ai_lungage_score = F.softmax(model(extracted_lung.to(device).unsqueeze(0).unsqueeze(0)).cpu().detach(), dim=1).numpy()[:, 1] 

    return ai_lungage_score

#################
#### return model with loaded weights in eval mode
#model = lungage_load()
##################