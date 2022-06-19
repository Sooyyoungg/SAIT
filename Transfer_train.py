# import os
# import sys
from skimage.metrics import mean_squared_error
from torch.optim import Adam
# sys.path.append("")
import torch
import matplotlib.pyplot as plt
import torchsummary

from Noise2Self.util import show, plot_images, plot_tensors, psnr
from Noise2Self.metric import frc, match_intensity, quantify, plot_quantifications
from Noise2Self.train import train
import Pix2Pix.model
from Transfer_Config import Transfer_Config
from Pix2Pix.Config import Config
from Pix2Pix.DataSplit import DataSplit
from Noise2Self.mask import Masker
from nc_loader import nc_loader
import copy as cp

config = Transfer_Config()
p2p_config = Config()
device = torch.device('cuda:' + str(config.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')

print("Training Start")
batch_size = 16
n_epoch = 10000 ## epoch 값
repeats = 1
learning_rate = 0.001
pretrained_name = 'SEM_best_epoch88_itr17160000_rmse2.7991442392364716.pt'
loss = 'mse' # [mse, mae]
Data_eval = {}
metrics_key = ['mse', 'ssim', 'frc']

## Prepare for self-supervision training
masker = Masker(width =4, mode='interpolate')

## if repeats

for repeat in range(repeats):
    print("Repeat No. {}".format(repeat))

    # data loader
    valid_data = DataSplit(data_list=config.valid_half_list, data_root=config.valid_root)
    print("len(valid_data) : ",len(valid_data))
    data_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True,
                                                    num_workers=16, pin_memory=False)

    [noisy, clean] = nc_loader(data_list=config.valid_half_list, data_root=config.valid_root).forward()
    # print(noisy.shape) # (50, 1, 66, 45)
    noisy = torch.Tensor(noisy)
    clean = torch.Tensor(clean)

    # model define
    model = cp.deepcopy(Pix2Pix.model.Pix2Pix(p2p_config))

    # transfer learning
    model.load_state_dict(torch.load(config.log_dir+'/'+pretrained_name)) # model directory
    model = cp.deepcopy(model.netG)
    # torchsummary.summary(model, (1, 66, 45), device='cpu')

    # optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    # training
    print("Training: model initialized with {} pretrained model".format)
    model = train(model, data_loader, loss, optimizer, n_epoch,
                  masker, earlystop=True, patience=10, device=device, verbose=True, ckpt_dir=config.ckpt_dir,
                  result_dir=config.result_dir, log_dir=config.t_log_dir)

    output = model(noisy)

    # plot example images
    if repeat == 0:
        nplot = 1
        plot_tensors([noisy[nplot, 0, :], clean[nplot, 0, :], output[nplot, 0, :]])

    output = output.cpu().detach().numpy()
    noisy = noisy.cpu().detach().numpy()
    clean = clean.cpu().detach().numpy()

    # RMSE 계산 (output & clean)
    mse = 0
    for i in range(output.shape[0]):
        mse += mean_squared_error(output[i, 0, :, :], clean[i, 0, :, :]) ** 0.5
    rmse = mse / output.shape[0]
    print("RMSE: ", rmse)
#
#     if repeat == 0 and True:
#         frc, spatial_freq = frc(output[0, 0, :], clean[0, 0, :])
#         plt.figure()
#         plt.plot(spatial_freq, frc, '-', linewidth=2, color='red', label='Pretrained with {}'.format(pretrained_name))
#         plt.legend(loc='lower left')
#         plt.title('FRC curve')
#
#     for sample in range(config['sample_size_list'][0]):
#         output[sample, :] = match_intensity(clean[sample, :], output[sample, :])
#
#     quantify(Data_eval, metrics_key, clean[0, 0, :], output[0, 0, :])
#
# plot_quantifications([Data_eval], ['Pretrained with FMD'], metrics_key,
#                      ylabel=['MSE', 'SSMI', 'Average FRC'], xlabel='Peak signal intensity (photon)',
#                      title='Microtubule test images: performance on peak signal intensity (# of shots: 10; loss: MSE)')