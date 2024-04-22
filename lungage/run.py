import torch
import torch.nn.functional as F

from .models import ai_lungage
from .preprocessing import lung_segmentation


def ai_lungage_score(folder_to_dcms, device = torch.device("cuda") ):

    """
    Predict AI lung age score given the path for series of dicoms of chest CT.

    Args:
        csv_path (str): Path to the folder of dcm series
    Returns:
        AI_Lung_Age_score --> 0 to 1, which 1 is most damage
    """
    
    lung_extracted = lung_segmentation(folder_to_dcms)
    model = ai_lungage()  # model with loaded weights in eval mode

    ai_lungage_score = F.softmax(model(lung_extracted.to(device).unsqueeze(0).unsqueeze(0)).cpu().detach(), dim=1).numpy()[:, 1] 

    return ai_lungage_score