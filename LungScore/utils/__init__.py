from .LungScoreCategories import  predict_riskgroup

#predict lung score risk category
def predict_lungscore_riskcategory(ai_lung_score):
    """
    Lung Score risk category
    Args:
        ai_lung_score (float): lung score predicted by the model
    Returns:
        ai_lung_score_risk_category (str): Lung score risk category 
    """
    ai_lung_score_risk_category = predict_riskgroup(ai_lung_score) 

    return ai_lung_score_risk_category