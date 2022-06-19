# Train Self-supervised network

import torch
from torch.nn import MSELoss, L1Loss
from torch.utils.tensorboard import SummaryWriter

def train(model,
          dataloader,
          loss,
          optimizer,
          epoch,
          masker,
          earlystop=True,
          patience=10,
          device="cuda:3",
          verbose=True,
          ckpt_dir = ckpt_dir,
          result_dir = result_dir
          ):
    index = 0 #iteration number
    val_loss, best_val_loss = 0.0, 200.0

    loss_switcher = {
        "mse": MSELoss(),
        "mae": L1Loss(),
    }
    loss_fn = loss_switcher.get(loss)

    iters_no_improve = 0

    for n in range(epoch):

        for i, data in enumerate(dataloader):
            noisy = data['sem'].to(device)
            clean = data['depth'].to(device)
            # train
            model.train()
            noisy, clean = (
                noisy.type(torch.cuda.FloatTensor),
                clean.type(torch.cuda.FloatTensor),
            )

            net_input, mask = masker.mask(noisy, index % (masker.n_masks - 1))
            # net_input: torch.Size([16, 1, 66, 45])

            optimizer.zero_grad()

            net_output = model(net_input)

            loss = loss_fn(net_output * mask, noisy * mask) * masker.n_masks

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
                        "iter [%d] : loss: %.5f | val_loss: %.5f"
                        % (index-1, loss.item(), val_loss.item())
                    )

            # tensorboard 저장하기
            t_noisy = fn_tonumpy(noisy)
            t_label = fn_tonumpy(label)
            t_input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            t_output = fn_tonumpy(fn_class(output))
            writer_train.add_image('input', t_noisy, epoch, dataformats='NHWC')
            writer_train.add_image('label', t_label, epoch, dataformats='NHWC')
            writer_train.add_image('masked input', t_input, epoch, dataformats='NHWC')
            writer_train.add_image('output', t_output, epoch, dataformats='NHWC')
            writer_train.add_scalar('loss', np.mean(loss), epoch)

            # 이미지 저장
            for j in range(t_label.shape[0]):
                id = j

                # png file
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), t_label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), t_input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), t_output[j].squeeze(), cmap='gray')

                # numpy type
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), t_label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), t_input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), t_output[j].squeeze())

    writer_train.close()

        if epoch % 10 == 0:
            save(ckpt_dir=ckpt_dir, net= model, optim=optim, epoch=epoch)
    
    return model
