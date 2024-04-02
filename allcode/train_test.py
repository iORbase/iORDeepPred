from train import train
from test import test
import shutil
import os

def train_test(target_model, load_model, training, predict, data_path, flag):
    type = target_model
    save_path = './lomi_npy/'
    if training:
        train(data_path + '.npy', flag, type, load_model=load_model)
    elif predict:
        test(data_path + '.npy', load_model)
