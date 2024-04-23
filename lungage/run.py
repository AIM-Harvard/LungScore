import torch
import torch.nn.functional as F

from .preprocessing import run_core # step 1: from dicom to NRRD
from .preprocessing import extract_lung # step 2: extract lung from NRRD
from .models import ai_lungage  # step 3: predict lung health score from 0 to 1, with 1 being the most damage

def ai_lungage_score(folder_to_dcms, device = torch.device("cuda") ):

    """
    Predict AI lung age score given the path for series of dicoms of chest CT.

    Args:
        folder_path (str): Path to the folder of dcm series
    Returns:
        AI_Lung_Age_score --> 0 to 1, which 1 is most damage
    """
    
    #step 1
    nrrd_scan = run_core(folder_to_dcms) 

    #step 2
    extracted_lung = extract_lung(nrrd_scan)

    #step 3
    model = ai_lungage()  # model with loaded weights in eval mode

    ai_lungage_score = F.softmax(model(extract_lung.to(device).unsqueeze(0).unsqueeze(0)).cpu().detach(), dim=1).numpy()[:, 1] 

    return ai_lungage_score