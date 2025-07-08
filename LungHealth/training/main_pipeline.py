"""
  Deep-learning biomarker for Lung Health training pipeline
"""

import os
import yaml
import argparse
import matplotlib  

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn import metrics
import numpy as np
import time
from torchmetrics import Accuracy
import wandb 
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics 
import pandas as pd
import monai
import torchvision.models as models

from LungHealth.datasets.dataset import Train_set, Tune_set
from LungHealth.models.model import Lunghealth
from LungHealth.training.training import train, tune

## ----------------------------------------
# path to config file
script_dir = os.path.dirname(os.path.abspath(__file__))
base_conf_file_path = os.path.join(script_dir, '..', '..', 'config')

conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']

parser = argparse.ArgumentParser(description = 'Run training pipeline')

parser.add_argument('--conf',
                    required = False,
                    help = 'Specify the YAML configuration file containing the run details. ' \
                            + 'Defaults to "training_pipeline.yaml"',
                    choices = conf_file_list,
                    default = "training_pipeline.yaml",
                   )


args = parser.parse_args()

conf_file_path = os.path.join(base_conf_file_path, args.conf)

with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

# input-output
training_data_folder_path = yaml_conf["io"]["path_to_data_folder_training"]
tuning_data_folder_path = yaml_conf["io"]["path_to_data_folder_tuning"]

path_to_csv_training =  yaml_conf["io"]["path_to_csv_training"]
path_to_csv_tuning =  yaml_conf["io"]["path_to_csv_tuning"]

# subdir under which the network weights should be saved
model_weights_foldertosave_name = yaml_conf["io"]["model_weights_foldertosave_name"]

# training / tuning / wandb
training_batch_size = yaml_conf["training"]["batch_size"]
num_epochs = yaml_conf["training"]["num_epochs"]
training_learningrate = yaml_conf["training"]["lr"]

conv_dropout = yaml_conf["training"]["conv_dropout"]
FC_dropout = yaml_conf["training"]["FC_dropout"]

normalization_value_min = yaml_conf["training"]["normalization_value_min"]
normalization_value_max = yaml_conf["training"]["normalization_value_max"]

tuning_batch_size = yaml_conf["tuning"]["batch_size"]

splits_of_classes = yaml_conf["wandb"]["splits of classes"]
normalization_method = yaml_conf["wandb"]["normalization_method"]
normalization_values = yaml_conf["wandb"]["normalization_values"]
Aim = yaml_conf["wandb"]["Aim"]

##########################################
# disturbed training 
model = nn.DataParallel(Lunghealth(conv_dropout, FC_dropout, normalization_value_min, normalization_value_max))

optimizer = torch.optim.Adam(model.parameters(), lr=training_learningrate)             
 
# setup training pipeline

dataset = Train_set(training_data_folder_path, path_to_csv_training)
data_loader = DataLoader(dataset, batch_size=training_batch_size, shuffle = True)

val_dataset = Tune_set(tuning_data_folder_path, path_to_csv_tuning)
tune_data_loader = DataLoader(val_dataset, batch_size=tuning_batch_size, shuffle = False)  

#######################


# run core
def main():
  
  
  Best_Tune_AUC = 0
  # wandb.config = dict( 
  #   epochs=num_epochs,
  #   classes=splits_of_classes,
  #   train_batch_size=training_batch_size,
  #   tune_batch_size=tuning_batch_size, 
  #   learning_rate=training_learningrate, 
  #   dropout_conv = conv_dropout,   
  #   dropout_fc = FC_dropout, 
  #   conv_filters = [16,32,64,128,256],  
  #   normalization = normalization_method,   
  #   normalizevalues = normalization_values, 
  #   Aim=Aim)    
     
  # wandb.init(project="AI_lung_health_q1nofindings_collate_fn_try", entity="ahmedadly98", config=wandb.config)    
  # wandb.run.name = "AI_lung_health_1"  
                
  for epoch in range(1, num_epochs):

      t = time.time() 
      # wandb.watch(model, log='all')
     
      # training AUC and loss
      train_loss, train_labels, train_logits = train(model, data_loader, optimizer)
      train_AUC =  roc_auc_score(train_labels, train_logits)   

      # tuning AUC and loss
      tune_loss, tune_labels, tune_logits = tune(model, tune_data_loader)
      tune_AUC =  roc_auc_score(tune_labels, tune_logits)     

      # save the model weights at each epoch
      torch.save(model.state_dict(), model_weights_foldertosave_name+'weights_atepoch_'+str(epoch))
    
      # save the best model based on AUC on tuning set   
      if tune_AUC > Best_Tune_AUC:  
        Best_Tune_AUC = tune_AUC
        torch.save(model.state_dict(), model_weights_foldertosave_name+'best_model_AUC_onTune_atepoch'+str(epoch))

      #wandb.log({"Epoch": epoch, "Tune_AUC": tune_AUC})  
      #wandb.log({"Epoch": epoch, "Tune_AUC": train_AUC})  

      #wandb.log({"epoch": epoch, "train_loss": train_loss}) 
      #wandb.log({"epoch": epoch, "tune_loss": tune_loss}) 

      #wandb.log({"epoch": epoch, "train_acc": train_acc})  
      #wandb.log({"epoch": epoch, "tune_acc": tune_acc})  

      print(f'Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Tune_Loss: {tune_loss:.4f}, time: {(time.time() - t):.4f}s')
      print(f'Epoch: {epoch:03d}, Train_AUC: {train_AUC:.4f}, , Tune_AUC: {tune_AUC:.4f}')
        

if __name__ == "__main__":
 
  print("\nTraining Starting.. ---\n")
  main()