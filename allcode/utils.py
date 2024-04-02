# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 21:13:03 2022

@author: Moore
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class InsectDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, index):
        
        data_tmp = self.x[index]
        
        y = np.array([np.float32(data_tmp['0'])])
        p = data_tmp['2']
        c = data_tmp['4']
        #label = data_tmp['3']
        #y = np.array([ tmp['0'] for tmp in data_tmp ])
        #y = np.float32(y)
        
        #p = np.array([ tmp['2'] for tmp in data_tmp ])
        #c = np.array([ tmp['4'] for tmp in data_tmp ])

        
        y = torch.from_numpy(y.astype(np.float32))
        p = torch.from_numpy(p.astype(np.float32))
        c = torch.from_numpy(c.astype(np.float32))

        return p,c,y        
        
        
    def __len__(self):
        return len(self.x)   
