import torch.nn as nn
import pandas as pd
import numpy as np
import imageio
from monai.transforms import ScaleIntensity, NormalizeIntensity
from torchvision import transforms

class DataSplit(nn.Module):
    def __init__(self, data_list, data_root, do_transform=False):
        super(DataSplit, self).__init__()

        self.data_list = pd.read_csv(data_list)
        self.do_transform = do_transform
        normal_transform = NormalizeIntensity(subtrahend=0.5, divisor=0.5, nonzero=False)
        scale_transform = ScaleIntensity(minv=-1.0, maxv=1.0)
        self.transform = transforms.Compose([normal_transform, scale_transform, transforms.ToTensor()])

        sems = []
        depths = []

        for i in range(len(self.data_list)):
            sem = imageio.imread(data_root+'/SEM/'+self.data_list.iloc[i, 0])
            sem = np.asarray(sem)   # (66, 45)
            sub = self.data_list.iloc[i, 0].split('_itr')[0]
            depth = np.asarray(imageio.imread(data_root+'/Depth/'+sub+'.png'))   # (66, 45)
            sems.append(sem)
            depths.append(depth)

        # Train: (40000, 66, 45) / Valid: (8000, 66, 45)
        self.sems = np.array(sems)
        self.depths = np.array(depths)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        sem = self.sems[item]
        depth = self.depths[item]

        # transform
        if self.do_transform is not None:
            sem = self.transform(sem)
            depth = self.transform(depth)

        return {"sem": sem, "depth": depth}
