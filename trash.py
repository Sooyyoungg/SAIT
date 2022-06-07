import pandas as pd
import numpy as np
import imageio

ll = np.asarray(imageio.imread('../SAIT_Data/Train/Depth/20201001_202940_NE142400C_RAE01_1_S01_M0005-01MS_3.png'))
print(ll.tolist())