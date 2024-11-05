import torch
import torch.nn.functional as F

from .preprocessing import dcm_to_nrrd #step 00: DICOM to NRRD
from .preprocessing import preprocess #step 0: preprocess NRRD, make sure of spacing , size, etc...
from .preprocessing import segment_lung #step 1: segment lung from NRRD
from .preprocessing import extract_lung # step 2: extract and preprocess lung from segmented lung NRRD

from .models import lungage_load # step 3: load model in eval mode with weights
from .models import lungage_predict #step 4: predict ai lung age score 
from .utils.risk_groups import predict_riskgroup #step 5: predict risk group based on lung age thresholds

def ai_lungage_score(dcm_path = 0, NRRD = 0):

    """
    Predict AI lung age score given the path for NRRD chest CT.

    Args:
        folder_path (str): Path to the nrrd scan
    Returns:
        AI_Lung_Age_score --> 0 to 1, which 1 is most damage
    """

    # if dicom then start with step 00 , if NRRD then start with step 0
    #step 00: dicom to nrrd
    nrrd = dcm_to_nrrd(dcm_path)
    
    #step 0: read nrrd and resample
    nrrd = preprocess(NRRD)

    #step 1
    segmented_lung = segment_lung(nrrd) 

    #step 2
    extracted_lung = extract_lung(segmented_lung, nrrd)

    #step 3
    model = lungage_load()

    #step 4: predict lung age from extracted lung using the loaded model
    ai_lungage_score = lungage_predict(model, extracted_lung)

    #step 5: predict risk group based on lung age thresholds 
    risk_group = predict_riskgroup(ai_lungage_score)

    return ai_lungage_score, risk_group