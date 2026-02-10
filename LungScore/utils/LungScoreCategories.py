"""
AI-derived Lung Score - Deciding Risk group based on lung score categories (5 risk groups)
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


def predict_riskgroup(ai_lung_score):
    """
    predict risk group based on lung score thresholds dervied from tuning set based on 5 risk groups
    0-25% - Very low risk group
    25%-50% - Low risk group
    50%-75% - Moderate risk group
    75%-95% - High risk group
    95%-100% - Very high risk group
    Args:
        ai_lung_score (int): (values from 0 to 1) representing lung score
    Returns:
        risk group category (str): representing risk group based on lung score
    """
    # lung score cut-offs calculated on tuning set based on lung damage score (1- lung score)
    ai_lung_score  =  1 - ai_lung_score
    
    if ai_lung_score <= 0.3239:     
        Lung_Score_category = "Group1: Very High Lung Score"
    elif (ai_lung_score > 0.3239) and (ai_lung_score <= 0.4345):   
        Lung_Score_category = "Group2: High Lung Score"
    elif (ai_lung_score > 0.4345) and (ai_lung_score <= 0.5478):   
        Lung_Score_category = "Group3: Moderate Lung Score" 
    elif (ai_lung_score > 0.5478) and (ai_lung_score <= 0.7123):  
        Lung_Score_category = "Group4: Low Lung Score"
    else:         
        Lung_Score_category = "Group5: Very Low Lung Score"

    return Lung_Score_category