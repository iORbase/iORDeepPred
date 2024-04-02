# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 21:37:50 2022

@author: Moore
"""





import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from sklearn import metrics
import numpy as np
import pandas as pd

import config



class dnn_model(pl.LightningModule):
    def __init__(self):
        super(dnn_model, self).__init__()
        
        self.fc1 = nn.Linear(1792,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512,512)
        self.fc5 = nn.Linear(512,1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)

        
    def forward(self, p,c):  
        
        x = torch.hstack((p,c))
        
        x = F.relu(self.bn1(self.fc1(x)))     
        x = F.relu(self.bn2(self.fc2(x)))    
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        
        x = self.fc5(x)
     
        return x
    
    def training_step(self, batch, batch_idx):

        p, c, y = batch
        logits = self.forward(p,c)
        loss = nn.MSELoss()(logits, y)
                
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        p, c, y, label, change_name  = batch
        logits = self.forward(p,c)
        val_loss = nn.MSELoss()(logits, y)

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return {'logits':logits,
                'label': label,
                'change_name':change_name}

    def test_step(self, batch, batch_idx):
        
        return self.validation_step(batch, batch_idx)
    
    def validation_epoch_end(self, outputs):
        #logits_all = torch.vstack([ x['logits'] for x in outputs ])
        #labels = torch.vstack([ x['label'] for x in outputs ])
        #print(outputs[0])
        
        logits_all = [ x['logits'] for x in outputs ]
        labels = [ x['label'] for x in outputs ]
        change_names = [ x['change_name'] for x in outputs ]
        #y_all = torch.vstack(i[ x['y'] for x in outputs])
        
        #logits_all = logits_all.cpu().numpy().squeeze()
        #y_all = y_all.cpu().numpy().squeeze()
        
        #logits_all = np.int8(logits_all <= 0)
        #y_all = np.int8(y_all <= 0)
       
        a = [ x[0] for x in  list(logits_all[0]) ]
        b = [ x.item() for x in a]
        
        a = list(labels[0])
        c = list(change_names[0])

        data = pd.DataFrame(np.concatenate(([c],[a],[b]),axis=0))
        data = data.T
        data.to_csv('test.csv')

        #acc = metrics.accuracy_score(y_all, logits_all)
        #print('predict result:')
        #with open('test.txt', 'w') as f:
        #    f.write(str(logits_all))
        #    f.write(str(labels))
         
        #print(labels)
        
        #self.log('val_acc', acc)
        #return acc

    def test_epoch_end(self, outputs):
        print(self.validation_epoch_end(outputs))
    
    def configure_optimizers(self):
            
        optimizer = optim.AdamW(lr=config.learning_rate, params=self.parameters())
               
      
        return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(
                    optimizer, mode='min', factor=config.scheduler_factor, patience=config.patience, verbose=True, min_lr=1e-5),
                'monitor': "val_loss"
                } 
