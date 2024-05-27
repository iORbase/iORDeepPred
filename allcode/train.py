# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from utils import InsectDataset
from model import dnn_model
import config

def train(data_path,flag,ckpt_name,load_model):
    x = np.load(data_path,allow_pickle=True)
    
    data_tr, data_val = train_test_split(x, test_size=0.1, random_state=config.seed)
    
    num_tr = len(data_tr)
    num_val = len(data_val)
    print("number of training samples:", num_tr)
    print("number of validation samples:", num_val)
    
    train_set = InsectDataset(x)
    val_set = InsectDataset(data_val)
    
    train_loader = DataLoader(dataset=train_set,batch_size=512,shuffle=True, num_workers=20)
    val_loader = DataLoader(dataset=val_set,batch_size=512,shuffle=False, num_workers=20)
    
    model = dnn_model()
    if flag == 1:
        model = model.load_from_checkpoint('./space/checkpoints/dnn/' + load_model + '.ckpt')
    
    ckpt_cb = ModelCheckpoint(
        monitor='val_loss', 
        mode='min', 
        dirpath='./space/checkpoints/'+config.model_type+'/', 
        filename=ckpt_name,
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
        gpus=config.gpus, 
        #precision=16,
        callbacks=Callbacks,
        logger=Logger,
        #strategy=config.strategy,
        num_sanity_val_steps=0,
        #fast_dev_run=True
        )
    
    trainer.fit(model=model,
                train_dataloaders=train_loader,val_dataloaders=val_loader)



