"""
  Deep-learning biomarker for Lung Health - Deciding Risk group based on lung health categories (5 risk groups)
"""
import numpy as np

def define_riskgroups_tuneset(tune_preds):
    """
    define 5 risk groups based on tuning set
    Args:
        tune_preds (numpy_array): numpy array with tuning set predictions (values from 0 to 1) representing lung age
    Returns:
        print statments with 5 risk groups thresholds
    """
    print("Very Low risk group <= ", np.quantile(tune_preds, 0.25)) 
    print("Low risk group <= ", np.quantile(tune_preds, 0.50))
    print("Intermediate risk group <= ", np.quantile(tune_preds, 0.75))
    print("High risk group <= ", np.quantile(tune_preds, 0.95))
    print("Very High risk group <= ", np.quantile(tune_preds, 1)) 

    return None


def predict_riskgroup(ai_lung_age_score):
    """
    predict risk group based on lung health thresholds from tuning set based on 5 risk groups
    Args:
        ai_lung_age_score (int): (values from 0 to 1) representing lung age
    Returns:
        risk group category
    """
    if ai_lung_age_score <= 0.3239:     
        risk_group_category = "Group1: Very low risk"
    elif (ai_lung_age_score > 0.3239) and (ai_lung_age_score <= 0.4345):   
        risk_group_category = "Group2: Low risk"
    elif (ai_lung_age_score > 0.4345) and (ai_lung_age_score <= 0.5478):   
        risk_group_category = "Group3: Intermediate risk" 
    elif (ai_lung_age_score > 0.5478) and (ai_lung_age_score <= 0.7123):  
        risk_group_category = "Group4: High risk"
    else:         
        risk_group_category = "Group5: Very High risk"

    return risk_group_category