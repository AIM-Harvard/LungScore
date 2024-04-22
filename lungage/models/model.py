import torch
import os
import torch.nn as nn

# Create CNN Model
class CNNModel(nn.Module): 
    def __init__(self, conv_dropout, FC_dropout, normalization_value_min, normalization_value_max): 
        super(CNNModel, self).__init__() 

        self.conv_dropout = conv_dropout   

        self.conv_layer1 = self._conv_layer_set1(1, 16, self.conv_dropout) 
        self.conv_layer2 = self._conv_layer_set234(16, 32, self.conv_dropout) 
        self.conv_layer3 = self._conv_layer_set234(32, 64, self.conv_dropout)  
        self.conv_layer4 = self._conv_layer_set234(64, 128, self.conv_dropout) 
        self.conv_layer5 = self._conv_layer_set5(128, 256, self.conv_dropout) 
        
        self.fc1 = nn.Linear(15360, 1024)  
        self.fc2 = nn.Linear(1024, 128) 
        self.fc3 = nn.Linear(128, 2)    
 
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=FC_dropout)  

        self.normalization_value_min = normalization_value_min
        self.normalization_value_max = normalization_value_max       
          
    def _conv_layer_set1(self, in_c, out_c, conv_dropout):  
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(7, 7, 7), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2,2,2)), 
        nn.BatchNorm3d(out_c),
        nn.Dropout(p=conv_dropout)
        ) 
        return conv_layer  
 
    def _conv_layer_set234(self, in_c, out_c, conv_dropout): 
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),  
        nn.LeakyReLU(),
        nn.MaxPool3d((2,2,2)),
        nn.BatchNorm3d(out_c),
        nn.Dropout(p=conv_dropout) 
        ) 
        return conv_layer 

    def _conv_layer_set5(self, in_c, out_c, conv_dropout): 
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),  
        nn.LeakyReLU(),
        nn.MaxPool3d((2,2,2), padding=(1,0,0)),
        nn.BatchNorm3d(out_c),
        nn.Dropout(p=conv_dropout) 
        ) 
        return conv_layer 

########################

    def NormalizeData(self, data):
         return (data - (self.normalization_value_min)) / ((self.normalization_value_max) - (self.normalization_value_min))    

    def forward(self, x):   

        out = self.NormalizeData(x)  
   
        out = self.conv_layer1(out)     
        out = self.conv_layer2(out) 
        out = self.conv_layer3(out) 
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)  
       
        out = out.view(out.size(0), -1)

        out = self.drop(self.relu(self.fc1(out)))
        out = self.drop(self.relu(self.fc2(out)))
        out = self.fc3(out) 

        return out