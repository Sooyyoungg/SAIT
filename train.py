import torch
import argparse
import pandas as pd
# import tensorboardX

from Config import Config
from DataSplit import DataSplit

config = Config()

## Data Loader
train_data = DataSplit(data_list=config.train_list, data_root=config.train_root)
valid_data = DataSplit(data_list=config.val_list, data_root=config.valid_root)

data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
data_loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
print(len(data_loader_train))

## Start Training
