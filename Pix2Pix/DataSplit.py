import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.image as img

class DataSplit(nn.Module):
    def __init__(self, data_list, data_root, do_transform=True):
        super(DataSplit, self).__init__()

        self.data_list = pd.read_csv(data_list)
        self.do_transform = do_transform
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

        tot_sem = []
        tot_depth = []

        for i in range(len(self.data_list)):
            # sem = np.full((66, 66), 167)
            sem_org = Image.open(data_root+'/SEM/'+self.data_list.iloc[i, 0]).convert('L')
            sem_org = np.asarray(sem_org)       # (66, 45)
            # print(sem_org[0, 0])
            # print(np.min(sem_org), np.max(sem_org))
            # sem[:, 10:55] = sem_org             # (66, 66)

            # plt.imshow(sem, cmap='gray')
            # plt.show()

            sub = self.data_list.iloc[i, 0].split('_itr')[0]
            # depth = np.full((66, 66), 167)
            depth_org = Image.open(data_root+'/Depth/'+sub+'.png')
            depth_org = np.asarray(depth_org)   # (66, 45)
            # depth[:, 10:55] = depth_org         # (66, 66)
            # print(np.min(depth_org), np.max(depth_org))

            tot_sem.append(sem_org)
            tot_depth.append(depth_org)

        # Train: (40000, 66, 45) / Valid: (8000, 66, 45)
        self.tot_sem = np.array(tot_sem)
        self.tot_depth = np.array(tot_depth)
        # (66, 45) (66, 45)
        # print(self.tot_sem[0].shape, self.tot_depth[0].shape)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        sem = self.tot_sem[item]
        depth = self.tot_depth[item]

        # transform
        if self.do_transform:
            sem = self.transform(sem)
            depth = self.transform(depth)

        return {"sem": sem, "depth": depth}
