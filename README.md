<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1><b>Lung Health</b></h1>
<p>Automated Lung Health Quantification.</p>
<img src="figures/Overview_FIGURE1.jpg" alt="Lung Health Pipeline" width="800" height="500"> 


<h2>Run the model</h2>
<p>To run the model on your dataset(s)</p>
<p>The model works on axial (LD)CT chest scans.</p>

    # Step 1: Install all our dependencies:
    pip install AI_lunghealth --pre

    # Step 2: Run this in your code environment
    from LungHealth.run import preprocess, extract_lung, lunghealth_load, lunghealth_predict, predict_lunghealth_riskcategory
    from lungmask import mask

    # step 3: preprocess nrrd and segment lung
    nrrd = preprocess(nrrd_path)
    lungmask = mask.apply(nrrd) 

    # step 4: extract and preprocess lung 
    extracted_lung = extract_lung(lungmask, nrrd)

    # step 5: load lunghealth model with weights
    model = lunghealth_load()

    # step 6: predict lung age from extracted lung using the loaded model
    ai_lunghealth_score = lunghealth_predict(model, extracted_lung)

    # step 7: predict risk group based on lung health thresholds (very low, low, moderate, high, very high)
    risk_group = predict_lunghealth_riskcategory(ai_lunghealth_score)


    # you can combine all in one step by:
    from LungHealth.run import AILunghealthscore
    ai_lunghealth_score, risk_group = AILunghealthscore(nrrd_path)


</body>
</html>



