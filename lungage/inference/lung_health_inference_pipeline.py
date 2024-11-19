"""
  Deep-learning biomarker for Lung Health testing pipeline
"""
from pathlib import Path
import wget
import os
import yaml
import argparse
import matplotlib
import numpy as np
import pandas as pd
import torch  
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from lungage.datasets.dataset import Test_set
from lungage.models.model import CNNModel

## ----------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_conf_file_path = os.path.join(script_dir, '..', '..', 'config')

conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']

parser = argparse.ArgumentParser(description = 'Run testing pipeline')

parser.add_argument('--conf',
                    required = False,
                    help = 'Specify the YAML configuration file containing the run details. ' \
                            + 'Defaults to "testing_pipeline.yaml"',
                    choices = conf_file_list,
                    default = "testing_pipeline.yaml",
                   )


args = parser.parse_args()

conf_file_path = os.path.join(base_conf_file_path, args.conf)

with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

"""
You can only change testing "path_to_data_folder_testing" and "csv_path_to_save_lung_health_scores".
Everything else should be kept the same.
"""

# input-output
testing_data_folder_path = yaml_conf["io"]["path_to_data_folder_testing"]

csv_path_to_save_lung_health_scores = yaml_conf["io"]["csv_path_to_save_lung_health_scores"]

# subdir under which the network weights are saved
model_weights_url = yaml_conf["io"]["model_weight_url_to_download"]

device_cuda = yaml_conf["io"]["device_cuda"]

# testing selection "nothing should be changed"
testing_batch_size = yaml_conf["testing"]["batch_size"]

normalization_value_min = yaml_conf["training"]["normalization_value_min"]
normalization_value_max = yaml_conf["training"]["normalization_value_max"]

conv_dropout = yaml_conf["training"]["conv_dropout"]
FC_dropout = yaml_conf["training"]["FC_dropout"]


##########################################

# setup testing pipeline

dataset = Test_set(testing_data_folder_path)
data_loader = DataLoader(dataset, batch_size=testing_batch_size, shuffle = False)

model = CNNModel(conv_dropout, FC_dropout, normalization_value_min, normalization_value_max)
device = torch.device(device_cuda)

###################################

# download model weights
def download_model_weights(model):
  
  #specificy working directory
  current_path = Path(os.getcwd())

  # download model weights in the specified working directory
  if not (current_path / "model_weights.pth").exists():
    wget.download(model_weights_url, out=os.path.join(os.getcwd(), 'model_weights.pth'))

    # Load the pretrained weights
  model = nn.DataParallel(model, device_ids = [0, 2, 3])
 
  model.load_state_dict(torch.load(current_path / "model_weights.pth", map_location=device))
    
  return model


#######################

# predict lung health score
def test(data_loader):
    
    scans = []
    scores = []
    
    # download and load model weights 
    model = download_model_weights(model = CNNModel(conv_dropout, FC_dropout, normalization_value_min, normalization_value_max))
    
    # model in evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            imgs, scan = batch
            pred = model.to(device)(imgs.to(device).unsqueeze(1))  
            
            # get a score between 0 to 1 , representing lung health
            pred = F.softmax(pred.cpu().detach(), dim=1).numpy()[:, 0] # 0 for the updated version of lung health , higher score --> better outcome  

            #print(scan)
            #print(pred)
            
            scan_id_name = os.path.splitext(scan)[0]
            scans.append(scan_id_name)
            scores.append(pred)

    ai_lung_health = pd.DataFrame(
    {'Scan': list(scans),
    'AI_Lung_Health_Score': list(scores)
    })
    ai_lung_health.to_csv(csv_path_to_save_lung_health_scores)

# run inference pipeline
def main():
    test(data_loader = data_loader)

if __name__ == "__main__":
 
  print("\nInference Pipeline Starting.. ---\n")
  main()