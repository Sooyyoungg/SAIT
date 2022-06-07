import torch
import argparse
import pandas as pd
# import tensorboardX

from Config import Config
from DataSplit import DataSplit
from model import Pix2Pix

config = Config()
device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

## Data Loader
train_list = pd.read_csv(config.train_list)

train_data = DataSplit(data_list=config.train_list, data_root=config.train_root)
valid_data = DataSplit(data_list=config.val_list, data_root=config.valid_root)

data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
data_loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
print(len(data_loader_train), "*", config.batch_size, "=", len(train_list))

## Start Training
model = Pix2Pix(config)

for epoch in range(config.n_epoch):
    for i, data in enumerate(data_loader_train):
        model.train()