import torch
import torch.nn.functional as F

from .preprocessing import preprocess_nrrd #step 1: preprocess NRRD, make sure of spacing , size, etc...
from .preprocessing import segment_lung #step 2: segment lung from NRRD
from .preprocessing import preprocess_lung # step 3: preprocess lung from segmented lung NRRD

from .models import lunghealth_load # step 4: load model in eval mode with weights
from .models import lunghealth_predict #step 5: predict ai lung score 
from .utils import predict_lunghealth_riskcategory #step 6: predict risk group based on lung score thresholds

def AILunghealthpredict(NRRD):

    """
    Predict AI lung score and risk group given the path for NRRD chest CT.

    Args:
        folder_path (str): Path to the nrrd scan
    Returns:
        ai_lunghealth_score (float): lung score from 0 to 1 -- 1 is most healthy
        risk group (str): lung score category -- very low (5), low (4), moderate (3), high (2), very high (1) 
    """
    
    # step 1: read nrrd and resample
    nrrd = preprocess_nrrd(NRRD)

    # step 2: segment the lung
    segmented_lung = segment_lung(nrrd) 

    # step 3: preprocessing the lung
    preprocessed_lung = preprocess_lung(segmented_lung, nrrd)

    # step 4: load the lung score model
    model = lunghealth_load()

    # step 5: predict lung score from extracted lung using the loaded model
    ai_lunghealth_score = lunghealth_predict(model, preprocessed_lung)

    # step 6: predict risk group based on lung score thresholds (1 to 5 -- very high, high, moderate, low, very low))
    risk_group = predict_lunghealth_riskcategory(ai_lunghealth_score)

    return ai_lunghealth_score, risk_group 