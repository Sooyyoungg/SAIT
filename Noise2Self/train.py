# Train Self-supervised network

import numpy as np
import os
import matplotlib as plt
import torch
from torch.nn import MSELoss, L1Loss
from torch.utils.tensorboard import SummaryWriter
from Noise2Self.util import *

def train(model,
          dataloader,
          loss,
          optimizer,
          n_epoch,
          masker,
          earlystop=True,
          patience=10,
          device="cuda:3",
          verbose=True,
          ckpt_dir=None,
          result_dir=None,
          log_dir=None
          ):
    index = 0 #iteration number
    val_loss, best_val_loss = 0.0, 1.0

    loss_switcher = {
        "mse": MSELoss(),
        "mae": L1Loss(),
    }
    loss_fn = loss_switcher.get(loss)

    iters_no_improve = 0

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    for epoch in range(n_epoch):

        for i, data in enumerate(dataloader):
            noisy = data['sem'].to(device)
            clean = data['depth'].to(device)
            # train
            model.train()
            noisy, clean = (
                noisy.type(torch.cuda.FloatTensor),
                clean.type(torch.cuda.FloatTensor),
            )

            net_input, mask = masker.mask(noisy, epoch % (masker.n_masks - 1))
            # net_input: torch.Size([16, 1, 66, 45])
            net_output = model(net_input)

            loss = loss_fn(net_output * mask, noisy * mask) # * masker.n_masks
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # tracing iteration number
            index += 1

            if index % 10 == 0: # validate
                model.eval()

                net_input, mask = masker.mask(noisy, masker.n_masks - 1)

                net_output = model(net_input)

                val_loss = loss_fn(net_output * mask, noisy * mask) * masker.n_masks

                # find best validation loss
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item() # update current best val loss
                    torch.save(model.state_dict(), ".best_checkpoint")
                    iters_no_improve = 0
                else:
                    iters_no_improve += 1
                    if iters_no_improve == patience:
                        print("early stopping initialized @ %d iterations." % (index))
                        model.load_state_dict(torch.load(".best_checkpoint"))
                        return model

                if verbose:
                    print(
                        "epoch [%d] iter [%d] : loss: %.5f | val_loss: %.5f"
                        % (epoch, index, loss.item(), val_loss.item())
                    )

            # tensorboard 저장하기
            t_noisy = fn_tonumpy(fn_denorm(noisy, mean=0.5, std=0.5))
            t_input = fn_tonumpy(fn_denorm(net_input, mean=0.5, std=0.5))
            t_output = fn_tonumpy(fn_denorm(net_output, mean=0.5, std=0.5))
            writer_train.add_image('input', t_noisy, epoch, dataformats='NHWC')
            writer_train.add_image('masked input', t_input, epoch, dataformats='NHWC')
            writer_train.add_image('output', t_output, epoch, dataformats='NHWC')
            writer_train.add_scalar('loss', loss, epoch)

            # 이미지 저장
            for j in range(t_output.shape[0]):
                id = j
                # png file
                plt.imsave(os.path.join(result_dir, 'input_%04d.png' % id), t_input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'output_%04d.png' % id), t_output[j].squeeze(), cmap='gray')

        if epoch % 10 == 0:
            save(ckpt_dir=ckpt_dir, net= model, optim=optimizer, epoch=epoch)

    writer_train.close()
    return model
