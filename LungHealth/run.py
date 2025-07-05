import torch
import torch.nn.functional as F

#from .preprocessing import dcm_to_nrrd #step 00: DICOM to NRRD
from .preprocessing import preprocess #step 0: preprocess NRRD, make sure of spacing , size, etc...
from .preprocessing import segment_lung #step 1: segment lung from NRRD
from .preprocessing import extract_lung # step 2: extract and preprocess lung from segmented lung NRRD

from .models import lunghealth_load # step 3: load model in eval mode with weights
from .models import lunghealth_predict #step 4: predict ai lung age score 
from .utils import predict_lunghealth_riskcategory #step 5: predict risk group based on lung age thresholds

def ai_lunghealth_score(NRRD):

    """
    Predict AI lung health score given the path for NRRD chest CT.

    Args:
        folder_path (str): Path to the nrrd scan
    Returns:
        ai_lunghealth_score --> 0 to 1 -- 1 is most healthy
    """

    # if dicom then start with step 0 , if NRRD then start with step 1
    # step 0: dicom to nrrd
    # nrrd = dcm_to_nrrd(dcm_path)
    
    # step 1: read nrrd and resample
    nrrd = preprocess(NRRD)

    # step 2: segment the lung
    segmented_lung = segment_lung(nrrd) 

    # step 3: preprocessing the lung
    extracted_lung = extract_lung(segmented_lung, nrrd)

    # step 4: load the lung health model
    model = lunghealth_load()

    # step 5: predict lung health from extracted lung using the loaded model
    ai_lunghealth_score = lunghealth_predict(model, extracted_lung)

    # step 6: predict risk group based on lung health thresholds (1 to 5 -- very high, high, moderate, low, very low))
    risk_group = predict_lunghealth_riskcategory(ai_lunghealth_score)

    return ai_lunghealth_score, risk_group 