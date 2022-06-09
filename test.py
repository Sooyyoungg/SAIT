import random
import torch
import pandas as pd
import tensorboardX
from torchvision.utils import save_image

from Config import Config
from DataSplit import DataSplit
from model import Pix2Pix
import networks

def main():
    config = Config()
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    print(device)

    ## Data Loader
    train_list = pd.read_csv(config.train_list)

    # train_data = DataSplit(data_list=config.train_list, data_root=config.train_root)
    valid_data = DataSplit(data_list=config.valid_list, data_root=config.valid_root)

    # data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
    data_loader_train = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
    print(len(data_loader_train), "x", config.batch_size,"(batch size) =", len(train_list))

    ## Start Training
    model = Pix2Pix(config)
    model.to(device)

    train_writer = tensorboardX.SummaryWriter(config.log_dir)

    print("Start Training!!")
    itr_per_epoch = len(data_loader_train)
    tot_itr = 0
    for epoch in range(config.n_epoch):
        for i, data in enumerate(data_loader_train):
            tot_itr += i
            train_dict = model.train(data)

            fake_depth = train_dict['fake_depth']
            real_depth = train_dict['real_depth']

            if i % 20 == 0:
                print("image save")
                r = random.randint(0, config.batch_size-1)
                # image 저장 및 RMSE 계산
                f_image = fake_depth[r]
                r_image = real_depth[r]
                # print(fake_depth, torch.min(fake_depth), torch.max(fake_depth)) -> [-1, 1]
                save_image(f_image, '{}/{}_{}_fake_depth.png'.format(config.img_dir, epoch+1, i+1))
                save_image(r_image, '{}/{}_{}_real_depth.png'.format(config.img_dir, epoch+1, i+1))

            # RMSE
            mse = 0
            for b in range(config.batch_size):
                diff = fake_depth[b] - real_depth[b]
                mse += torch.pow(diff, 2)
            rmse = torch.sqrt(mse / config.batch_size)

            # save & print loss values
            train_writer.add_scalar('Loss_G', train_dict['G_loss'])
            train_writer.add_scalar('Loss_D', train_dict['D_loss'])
            print("Epoch: %d/%d | itr: %d/%d | tot_itrs: %d | Loss_G: %.9f | Loss_D: %.9f | RMSE: %.9f"%(epoch+1, config.n_epoch, i+1, itr_per_epoch, tot_itr, train_dict['G_loss'], train_dict['D_loss'], rmse))


if __name__ == '__main__':
    main()