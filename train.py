import torch
import pandas as pd
# import tensorboardX

from Config import Config
from Data.DataSplit import DataSplit
from model import Pix2Pix
import networks

config = Config()
device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

## Data Loader
train_list = pd.read_csv(config.train_list)

train_data = DataSplit(data_list=config.train_list, data_root=config.train_root)
valid_data = DataSplit(data_list=config.val_list, data_root=config.valid_root)

data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
data_loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
print(len(data_loader_train), "x", config.batch_size, "=", len(train_list))

## Start Training
model = Pix2Pix(config)

print("Start Training!!")
itr_per_epoch = len(data_loader_train) / config.batch_size
tot_itr = 0
for epoch in range(config.n_epoch):
    for i, data in enumerate(data_loader_train):
        tot_itr += i
        train_dict = model.train(data)

        fake_depth = train_dict['fake_depth']
        real_depth = train_dict['real_depth']

        print("Epoch[%d/%d] | itr[%d/%d] | tot_itrs: %d | Loss_G: %.9f | Loss_D: %.9f".format(epoch, config.n_epoch, i, itr_per_epoch, tot_itr, train_dict['G_loss'], train_dict['D_loss']))

    valid_G_loss = 0
    valid_D_loss = 0
    for v, v_data in enumerate(data_loader_valid):
        val_dict = model.val(v_data)
        valid_G_loss += val_dict['G_loss']
        valid_D_loss += val_dict['D_loss']
    v_G_avg_loss = float(valid_G_loss / (v+1))
    v_D_avg_loss = float(valid_D_loss / (v+1))
    print("===> Validation <=== Epoch[%d/%d] | Loss_G: %.9f | Loss_D: %.9f".format(epoch, config.n_epoch, v_G_avg_loss, v_D_avg_loss))

    networks.update_learning_rate(model.G_scheduler, model.optimizer_G)
    networks.update_learning_rate(model.D_scheduler, model.optimizer_D)

