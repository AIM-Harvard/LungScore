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
<p>To run Lung Health on you dataset</p>
<p>The model works on axial (LD)CT chest scans.</p>

    # Step 1: Install all our dependencies:
    pip install AIlunghealth --pre

    # Step 2: Import Lung health functions
    from LungHealth.run import preprocess_nrrd, segment_lung, preprocess_lung, lunghealth_load, lunghealth_predict, predict_lunghealth_riskcategory

    # step 3: preprocess nrrd and segment the lung by passing nrrd_file_path --ex: nrrd_path="/mnt/data/123img.nrrd"
    nrrd = preprocess_nrrd(nrrd_path)
    lungmask = segment_lung(nrrd) 

    # step 4: preprocess lung 
    preprocessed_lung = preprocess_lung(lungmask, nrrd)

    # step 5: load lunghealth model weights
    model = lunghealth_load()

    # step 6: predict lung health score (score from 0 t0 1 -- 1 is healthiest lung)
    ai_lunghealth_score = lunghealth_predict(model, preprocessed_lung)

    # step 7: predict risk group based on lung health splits (very low, low, moderate, high, very high)
    risk_group = predict_lunghealth_riskcategory(ai_lunghealth_score)

    # you can combine all in one step by:
    from LungHealth.run import AILunghealthpredict
    ai_lunghealth_score, risk_group = AILunghealthpredict(nrrd_path)


</body>
</html>

