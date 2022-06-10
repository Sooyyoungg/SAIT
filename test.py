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
    test_list = pd.read_csv(config.test_list)

    test_data = DataSplit(data_list=config.test_list, data_root=config.test_root)

    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)
    print("Test: ", len(data_loader_test), "x", config.batch_size,"(batch size) =", len(test_list))

    ## Start Training
    model = Pix2Pix(config)
    model.load_state_dict(torch.load(config.log_dir))
    model.to(device)
    model.eval()

    print("Start Testing!!")
    tot_itr = 0
    mse = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader_test):
            tot_itr += i
            test_dict = model.test(data)

            fake_depth = test_dict['fake_depth']
            real_depth = test_dict['real_depth']

            # image 저장
            save_image(fake_depth, '{}/{}_{}_fake_depth.png'.format(config.test_img_dir,))
            save_image(real_depth, '{}/{}_{}_real_depth.png'.format(config.test_img_dir, epoch+1, i+1))

            # RMSE 계산
            diff = fake_depth - real_depth
            mse += torch.pow(diff, 2)

            # print loss values
            print("Loss_G: %.9f | Loss_D: %.9f"%(test_dict['G_loss'], test_dict['D_loss']))

        rmse = torch.sqrt(mse / len(data_loader_test))
        print("==> Testing <== Avg Loss_G: %.9f | Avg Loss_D: %.9f | RMSE: %.9f"%(test_dict['G_loss'], test_dict['D_loss'], rmse))


if __name__ == '__main__':
    main()