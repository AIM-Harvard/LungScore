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


def predict_riskgroup(ai_lung_health_score):
    """
    predict risk group based on lung health thresholds from tuning set based on 5 risk groups
    Args:
        ai_lung_health_score (int): (values from 0 to 1) representing lung health
    Returns:
        risk group category
    """
    #to get old score pipeline
    ai_lung_health_score  =  1 - ai_lung_health_score
    
    if ai_lung_health_score <= 0.3239:     
        Lung_Health_category = "Group1: Very High Lung Health"
    elif (ai_lung_health_score > 0.3239) and (ai_lung_health_score <= 0.4345):   
        Lung_Health_category = "Group2: High Lung Health"
    elif (ai_lung_health_score > 0.4345) and (ai_lung_health_score <= 0.5478):   
        Lung_Health_category = "Group3: Moderate Lung Health" 
    elif (ai_lung_health_score > 0.5478) and (ai_lung_health_score <= 0.7123):  
        Lung_Health_category = "Group4: Low Lung Health"
    else:         
        Lung_Health_category = "Group5: Very Low Lung Health"

    return Lung_Health_category