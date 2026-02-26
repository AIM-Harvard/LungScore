<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1><b>AI-based Radiographic Lung Score Associates with Clinical Outcomes in Adults</b></h1>
<p>This repository contains the code and the evaluation of the LungScore (an AI-based Radiographric Score of your Lung Integrity).</p>
<p>Automated Lung Score Quantification.</p>
<img src="figures/Overview_LungScore.jpg" alt="Lung Score Pipeline" width="800" height="500"> 


<h2>Repository Structure</h2>
<p>This repository is structured as follows:</p>


<h3>Run the model</h3>
<p>To run Lung Score on you dataset</p>
<p>The model works on axial (LD)CT chest scans.</p>

    # Step 1: Install all our dependencies:
    pip install LungScore --pre

    # Step 2: Import Lung score functions
    from LungScore.run import preprocess_nrrd, segment_lung, preprocess_lung, lungscore_load, lungscore_predict, predict_lungscore_riskcategory

    # step 3: preprocess nrrd and segment the lung by passing nrrd_file_path --ex: nrrd_path="/mnt/data/123img.nrrd"
    nrrd = preprocess_nrrd(nrrd_path)
    lungmask = segment_lung(nrrd) 

    # step 4: preprocess lung 
    preprocessed_lung = preprocess_lung(lungmask, nrrd)

    # step 5: load Lung Score model weights
    model = lungscore_load()

    # step 6: predict Lung Score (score from 0 t0 1 -- 1 is least impaired lung)
    ai_lung_score = lungscore_predict(model, preprocessed_lung)

    # step 7: predict risk group based on Lung Score splits (very low, low, moderate, high, very high)
    risk_group = predict_lungscore_riskcategory(ai_lung_score)

    # you can combine all in one step by:
    from LungScore.run import AILungscorepredict
    ai_lung_score, risk_group = AILungscorepredict(nrrd_path)


<h4>Disclaimer</h4>
<p>The code and data of this repository are provided to promote reproducible research. They are not intended for clinical care or commercial use.</p>
<p>The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.</p>


</body>
</html>

