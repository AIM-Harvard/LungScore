import torch
import torch.nn as nn
from torchmetrics import Accuracy
import time 
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0")

def train(model, data_loader, optimizer):

    train_loss = 0      
    train_acc = 0      

    train_labels = np.array([])   
    train_logits = np.array([])     

    accuracy = Accuracy(task="binary").to(device)
    crit = nn.CrossEntropyLoss() 

    model.train()
    for batch in data_loader:

        imgs, labels = batch 
        labels = labels.cuda()
 
        optimizer.zero_grad() 

        preds = model.to(device)(imgs.to(device).unsqueeze(1))  
      
        loss = crit(preds, labels)  
    
        train_loss += loss 

        pred = torch.argmax(preds, dim=1)    
        acc = accuracy(pred, labels) 
      
        train_logits = np.append(train_logits, F.softmax(preds.cpu().detach(), dim=1).numpy()[:, 1]) 
        train_labels = np.append(train_labels, (labels.cpu().detach().numpy()))
 
        train_acc += acc  
      
        loss.backward()
        optimizer.step()

    return train_loss, train_acc, train_labels, train_logits


def tune(model, data_loader):
    tune_loss = 0   
    tune_acc = 0        
 
    accuracy = Accuracy(task="binary").to(device)
    crit = nn.CrossEntropyLoss() 

    tune_labels = np.array([])   
    tune_logits = np.array([])  

    model.eval()
    with torch.no_grad():
         for batch in data_loader:
             imgs, labels = batch
             labels = labels.to(device)
              
             preds = model.to(device)(imgs.to(device).unsqueeze(1))  
             
             loss = crit(preds, labels)  
             tune_loss += loss
             
             tune_logits = np.append(tune_logits, F.softmax(preds.cpu().detach(), dim=1).numpy()[:, 1]) 
             tune_labels = np.append(tune_labels, (labels.cpu().detach().numpy()))
 
             preds_argmax = torch.argmax(preds, dim=1) 
             acc = accuracy(preds_argmax, labels) 
 
             tune_acc += acc 

    return tune_loss, tune_acc, tune_labels, tune_logits



