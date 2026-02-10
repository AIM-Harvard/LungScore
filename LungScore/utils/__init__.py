from .LungHealthCatefgories import  predict_riskgroup

#predict lung health risk category
def predict_lunghealth_riskcategory(ai_lung_health_score):
    """
    Lung Health risk category
    Args:
        ai_lung_health_score (float): score of lung health predicted by the model
    Returns:
        ai_lung_health_risk_category (str): Lung health category 
    """
    ai_lung_health_risk_category = predict_riskgroup(ai_lung_health_score) 

    return ai_lung_health_risk_category
