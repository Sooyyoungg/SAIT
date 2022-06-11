import os
import sys

import Pix2Pix.model

sys.path.append("..")

import torch
import math
import json
import matplotlib.pyplot as plt
import os.path
import datetime
import numpy as np

from util import show, plot_images, plot_tensors, psnr
from data_loader import tiff_loader, load_confocal

from util import getbestgpu
from models.unet import Unet
from metric import frc, match_intensity, quantify, plot_quantifications
from train import train
from torch.optim import Adam

device = torch.device('cuda:' + str(self.config.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')

print("Training Start")
n_epoch = 100 ## epoch 값
repeats = 10
learning_rate = 0.0001
pretrained_name = 'fmd_epoch50.pt'
loss = 'mse' # [mse, mae]
key_s = 10
Data_eval[key_s] = {}
metrics_key = ['mse', 'ssmi', 'frc']

## Prepare for self-supervision training
from mask import Masker
masker = Masker(width =4, mode='interpolate')

## if repeats

for repeat in range(repeats):
    print("No. {}".format(repeat))

    # data loader
    train_data = DataSplit(data_list=config.train_list, data_root=config.train_root)
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                                    num_workers=16, pin_memory=False)
    noisy = data_loader['sem']
    clean = data_loader['depth']

    # model define
    model = Pix2Pix.model.Pix2Pix().netG

    # transfer learning
    model.load_state_dict(torch.load(config.lod_dir+'/'+pretrained_name)) # model directory

    # optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    # training
    print("Training: model initialized with {} pretrained model".format)
    model = train(model, data_loader, loss, optimizer, n_epoch, \
                      masker, earlystop=True, patience=10, device=device, verbose=True)

    output = model(noisy)

    # plot example images
    if repeat == 0:
        nplot = 1
        plot_tensors([noisy[nplot, 0, :], clean[nplot, 0, :], output[nplot, 0, :]])

    output = output.cpu().detach().numpy()
    noisy = noisy.cpu().detach().numpy()
    clean = clean.cpu().detach.numpy()

    if repeat == 0 and True:
        frc, spatial_freq = frc(output[0, 0, :], clean[0, 0, :])
        plt.figure()
        plt.plot(spatial_freq, frc, '-', linewidth=2, color='red', label='Pretrained with {}'.format(pretrained_name))
        plt.legend(loc='lower left')
        plt.title('FRC curve')

    for sample in range(config['sample_size_list'][0]):
        output[sample, :] = match_intensity(clean[sample, :], output[sample, :])

    quantify(Data_eval[key_s], metrics_key, clean[0, 0, :], output[0, 0, :])

plot_quantifications([Data_eval], ['Pretrained with FMD'], metrics_key,
                     ylabel=['MSE', 'SSMI', 'Average FRC'], xlabel='Peak signal intensity (photon)',
                     title='Microtubule test images: performance on peak signal intensity (# of shots: 10; loss: MSE)')