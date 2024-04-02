# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from utils_test import InsectDataset
from model_test import dnn_model
import config

def test(data_path,ckpt_name):
    x = np.load(data_path,allow_pickle=True)
    
    #data_tr, data_val = train_test_split(x, test_size=0.01, random_state=config.seed)
    #data_tr, data_val = train_test_split(x, test_size=1)    

    #num_tr = len(data_tr)
    #num_val = len(data_val)
    #print("number of training samples:", num_tr)
    #print("number of validation samples:", num_val)
    
    #train_set = InsectDataset(data_tr)
    #val_set = InsectDataset(data_val)
    test_set = InsectDataset(x)
    
    #train_loader = DataLoader(dataset=train_set,batch_size=512,shuffle=True, num_workers=20)
    #val_loader = DataLoader(dataset=val_set,batch_size=512,shuffle=False, num_workers=20)
    test_loader = DataLoader(dataset=test_set,batch_size=len(x),shuffle=False, num_workers=20)
    
    model = dnn_model()
    model = model.load_from_checkpoint('./space/checkpoints/dnn/'+ ckpt_name +'.ckpt')
    
    ckpt_cb = ModelCheckpoint(
        monitor='val_loss', 
        mode='min', 
        dirpath='./space/checkpoints/'+config.model_type+'/', 
        filename='best',
        save_last=False,
        )
    
    es = EarlyStopping(
        monitor='val_loss', 
        patience=config.patience_stop, 
        mode='min',
        )
    
    Logger = TensorBoardLogger(
        save_dir='./space/logs/', 
        name=config.model_type,
        )
    
    Callbacks = [es, ckpt_cb]
    
    trainer = pl.Trainer(
        max_epochs=config.epochs_max,
        devices=1,
        #gpus=config.gpus, 
        #precision=16,
        #callbacks=Callbacks,
        #logger=Logger,
        strategy=config.strategy,
        num_sanity_val_steps=0,
        # fast_dev_run=True
        )
    trainer.test(model=model,dataloaders=test_loader)

