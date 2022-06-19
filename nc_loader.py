import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

class nc_loader(nn.Module):
    def __init__(self, data_list, data_root, do_transform=True):
        super(nc_loader, self).__init__()

        self.data_list = pd.read_csv(data_list)
        self.do_transform = do_transform
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

        tot_sem = []
        tot_depth = []
        # g_root = '/scratch/connectome/conmaster/Pycharm_projects/SAIT/Pix2Pix/Generated_images/2layers'

        for i in range(len(self.data_list)):
            sem = Image.open(data_root+'/SEM/'+self.data_list.iloc[i, 0]).convert('L')
            sem = np.reshape(np.asarray(sem), (1, 66, 45))   # (66, 45)
            sub = self.data_list.iloc[i, 0].split('_itr')[0]
            depth = np.reshape(np.asarray(Image.open(data_root+'/Depth/'+sub+'.png').convert('L')), (1, 66, 45))   # (66, 45)
            tot_sem.append(sem)
            tot_depth.append(depth)

        # Train: (40000, 66, 45) / Valid: (8000, 66, 45)
        self.tot_sem = np.array(tot_sem)
        self.tot_depth = np.array(tot_depth)

    def forward(self):
        return  [self.tot_sem, self.tot_depth]
