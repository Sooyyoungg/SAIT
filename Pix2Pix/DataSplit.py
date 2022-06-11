import torch.nn as nn
import pandas as pd
import numpy as np
import imageio
from torchvision import transforms

class DataSplit(nn.Module):
    def __init__(self, data_list, data_root, do_transform=False):
        super(DataSplit, self).__init__()

        self.data_list = pd.read_csv(data_list)
        self.do_transform = do_transform
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

        tot_sem = []
        tot_depth = []

        for i in range(len(self.data_list)):
            sem = imageio.imread(data_root+'/SEM/'+self.data_list.iloc[i, 0])
            sem = np.asarray(sem)   # (66, 45)
            sub = self.data_list.iloc[i, 0].split('_itr')[0]
            depth = np.asarray(imageio.imread(data_root+'/Depth/'+sub+'.png'))   # (66, 45)
            tot_sem.append(sem)
            tot_depth.append(depth)

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
        if self.do_transform is not None:
            sem = self.transform(sem)
            depth = self.transform(depth)

        return {"sem": sem, "depth": depth}
