"""
  Deep-learning biomarker for Lung Health - Deciding Risk group based on lung health categories (5 risk groups)
"""
import numpy as np

def define_riskgroups_tuneset(tune_logits):
    """
    define 5 risk groups based on tuning set
    Args:
        tune_logits (numpy_array): numpy array with tuning set predictions logits (values from 0 to 1) representing lung age
    Returns:
        print statments with 5 risk groups thresholds
    """
    print("Very Low risk group <= ", np.quantile(tune_logits, 0.25)) 
    print("Low risk group <= ", np.quantile(tune_logits, 0.50))
    print("Intermediate risk group <= ", np.quantile(tune_logits, 0.75))
    print("High risk group <= ", np.quantile(tune_logits, 0.95))
    print("Very High risk group <= ", np.quantile(tune_logits, 1)) 

    return None


def predict_riskgroup(logit):
    """
    predict risk group based on lung health thresholds from tuning set based on 5 risk groups
    Args:
        logit (int): (values from 0 to 1) representing lung age
    Returns:
        risk group category
    """
    if logit <= 0.3239:     
        risk_group_category = "Group1: Very low risk"
    elif (logit > 0.3239) and (logit <= 0.4345):   
        risk_group_category = "Group2: Low risk"
    elif (logit > 0.4345) and (logit <= 0.5478):   
        risk_group_category = "Group3: Intermediate risk" 
    elif (logit > 0.5478) and (logit <= 0.7123):  
        risk_group_category = "Group4: High risk"
    else:         
        risk_group_category = "Group5: Very High risk"

    return risk_group_category