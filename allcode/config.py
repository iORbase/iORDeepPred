# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:31:12 2022

@author: Moore
"""



seed = 1234

model_type = 'dnn'

epochs_max = 50
gpus = [7]

patience = 5                  # 学习率裁剪的epoch数

strategy = 'ddp'

learning_rate = 1e-4                # 初始学习率
scheduler_factor = 0.5

patience_stop = 20                  # 训练提前终止的epoch数
