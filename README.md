<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1><b>Lung Age</b></h1>
<p>Automated Lung Health Quantification.</p>
<!-- img src="fig1.jpg" width="104" height="142" -->

<h2>Repository Structure</h2>

<p>The Lung Age repository is structured as follows:</p>

<ul>
<li><p>All the source code to run the deep-learning-based fully automatic lung health quantification pipeline is found under the lungage folder.</p></li>
<li><p>Models weights necessary to run the pipeline, are stored under the data folder.</p></li>
<li><p>Statistical analysis are located in the stats_analysis folder.</p></li>
</ul>

<h2>Run the model</h2>
<p>To run the model on your dataset(s)</p>
<p>The model can work on axial (LD)CT chest and cardiac scans where the whole or part of the lung is available, however for better quantification, whole chest CT is required.</p>

    #Step 1: Install all our dependencies:
    pip install AI_LUNG_HEALTH --pre

    #Step 2: Run this in your code environment, -GPU is needed-
    from lungage.run import ai_lungage_score

    # step 3: predict AI_lung_Health_Score by passing path to your DICOM folder for one instance - ex: '/mnt/data/mydicom/'
    ai_lungage_score = ai_lungage_score(folder_to_dcms)

<p>You can also run the model on your dataset(s) in Mhub</p>

    #to do
    #to do

<h2>To train Lung Age</h2>
    #use the training pipeling -- to do


<h2>Cite</h2>

</body>
</html>



