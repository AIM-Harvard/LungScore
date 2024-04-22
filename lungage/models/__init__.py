import os
import pickle
from pathlib import Path

import torch
import wget
from .model import lungage

def ai_lungage(eval_mode=True, device = torch.device('cuda')):

    model = lungage().to(device)

    weights_url = "https://zenodo.org/records/10956969?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjUyYjIwMjQ[…]py73xZfqy9pb5GtHSZ43v29MOg5tJ0BRFQnl4kZ_khP5oN87pA8scA1OEW3jPFQ"
    current_path = Path(os.getcwd())

    if not (current_path / "model_weights.torch").exists():
        wget.download(weights_url, bar=bar_progress)

    # Load the pretrained weights
    model.load_state_dict(torch.load(current_path / "model_weights.torch", map_location=device))

    if eval_mode:
        model.eval()

    return model