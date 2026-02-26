<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1><b>AI-based Radiographic Lung Score Associates with Clinical Outcomes in Adults</b></h1>
<p>Repository for "LungScore": An AI-based Radiographric Score of Lung Integrity applicable to all adults including non-smokers and those without overt disease.</p>
<img src="figures/Overview_LungScore.jpg" alt="Lung Score Overview" width="800" height="500"> 


<h2>Repository Structure</h2>
<p>This repository is structured as follows:</p>

<h2>Environment Setup</h2>
<p></p>

<h2>Run the model</h2>
<p>To run Lung Score on you dataset</p>
<p>The model works on axial chest (LD)CT scans.</p>

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


<h2>Replicating Lung Score Pipeline</h2>
<h3>Training Pipeline</h3>
<p></p>

<h3>Inference Pipeline</h3>
<p></p>

<h2>Lung Score Model</h2>
<h3>Model training and Network Architecture</h3>
<p></p>
<img src="figures/LungScore_pipeline.jpg" alt="Lung Score Pipeline" width="550" height="500"> 



<h3>Model Validation</h3>
<p></p>

<h2>Statistical Analysis Code</h2>
<p></p>

<h2>Datasets</h2>
<p>
  The LungScore was trained on the <strong>National Lung Screening Trial (NLST)</strong> and tested on a held-out test set from NLST and an external dataset from the <strong>Framingham Heart Study (FHS)</strong>. These datasets can be requested through official repositories as follows:
</p>
<ul>
  <li><strong>National Lung Screening Trial (NLST):</strong><a href="https://biometry.nci.nih.gov/cdas/nlst/">NCI CDAS</a>.</li>
  <li><strong>Framingham Heart Study (FHS):</strong><a href="https://biolincc.nhlbi.nih.gov/">BioLINCC</a>.</li>
</ul>


<h2>Disclaimer</h2>
<p>The code and data of this repository are provided to promote reproducible research. They are not intended for clinical care or commercial use.</p>
<p>The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.</p>

<h2>Contact</h2>
<p>We are happy to help you. Any question regarding this repository, please reach out to ahassan12@bwh.harvard.edu and haerts@bwh.harvard.edu.</p>


</body>
</html>

