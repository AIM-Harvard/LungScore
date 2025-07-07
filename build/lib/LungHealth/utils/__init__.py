from .LungHealthCatefgories import  predict_riskgroup

#predict lung health risk category
def predict_lunghealth_riskcategory(ai_lung_health_score):
    """
    Lung Health risk category
    Args:
        ai_lung_health_score: score of lung health predicted from the model
    Returns:
        ai_lung_health_risk_category: Lung health risk category 
    """
    ai_lung_health_risk_category = predict_riskgroup(ai_lung_health_score) 

    return ai_lung_health_risk_category
