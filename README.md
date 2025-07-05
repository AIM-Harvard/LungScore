<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1><b>Lung Health</b></h1>
<p>Automated Lung Health Quantification.</p>
<img src="figures/overview_figure_1_lung_health.jpg" alt="Lung Health Pipeline" width="800" height="500"> 

<h2>Repository Structure</h2>

<p>The Lung Health repository is structured as follows:</p>

<ul>
<li><p>All the source code to run the deep-learning-based fully automatic lung health quantification pipeline is found under the Lung health folder.</p></li>
<li><p>Models weights necessary to run the pipeline, can be downloaded from zenodo - see next section -.</p></li>
<li><p>Statistical analysis are located in the stats_analysis folder.</p></li>
</ul>

<h2>Run the model</h2>
<p>To run the model on your dataset(s)</p>
<p>The model works on axial (LD)CT chest scans.</p>

    #Step 1: Install all our dependencies:
    pip install AI_lungage --pre

    #Step 2: Run this in your code environment
    from lungage.run import dcm_to_nrrd, preprocess, extract_lung, lungage_load, lungage_predict
    from lungmask import mask

    #step 3: if Dicom, else jump to next step to NRRD
    #nrrd = dcm_to_nrrd(dcm_path)

    #step 3: preprocess nrrd and segment lung
    nrrd = preprocess(nrrd_path)
    lungmask = mask.apply(nrrd) 

    #step 4: extract and preprocess lung
    extracted_lung = extract_lung(lungmask, nrrd)

    #step 5: load lungage model with weights
    model = lungage_load()

    #step 6: predict lung age from extracted lung using the loaded model
    ai_lungage_score = lungage_predict(model, extracted_lung)

    #step 7: predict risk group based on lung age thresholds 
    risk_group = predict_riskgroup(ai_lungage_score)


    # you can combine all in one step by:
    from lungage.run import ai_lungage_score
    ai_lungage_score, risk_group = ai_lungage_score(nrrd)


</body>
</html>



