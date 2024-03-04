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

from dataset import Train_set, Tune_set
from model import CNNModel

## ----------------------------------------

base_conf_file_path = 'config/'
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

path_to_labels_training =  yaml_conf["io"]["path_to_labels_training"]
path_to_labels_tuning =  yaml_conf["io"]["path_to_labels_tuning"]

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

# setup training pipeline

dataset = Train_set(training_data_folder_path, path_to_labels_training)
data_loader = DataLoader(dataset, batch_size=training_batch_size, shuffle = True)

val_dataset = Tune_set(tuning_data_folder_path, path_to_labels_tuning)
val_data_loader = DataLoader(val_dataset, batch_size=tuning_batch_size, shuffle = False)  
 
# disturbed training 
net = nn.DataParallel(CNNModel(conv_dropout, FC_dropout, normalization_value_min, normalization_value_max), device_ids = [0, 1, 2])

opt = torch.optim.Adam(net.parameters(), lr=training_learningrate)             
 
device = torch.device("cuda:0")
x = torch.rand(1,1,90,280,400).to(device)
print(net(x))
#######################


# run core