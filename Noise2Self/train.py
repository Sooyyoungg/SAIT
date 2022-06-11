# Train Self-supervised network

import torch
from torch.nn import MSELoss, L1Loss

def train(model,
          dataloader,
          loss,
          optimizer,
          epoch,
          masker,
          earlystop=True,
          patience=10,
          device="cuda:0",
          verbose=True
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
                print(val_loss.item())

                # find best validation loss
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item # update current best val loss
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
    return model
