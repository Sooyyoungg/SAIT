import pandas as pd
import numpy as np
import imageio
import glob

# make data list
val_file_dir = './Dataset/Validation/SEM/*'
val_files = sorted(glob.glob(val_file_dir))

val_file_list = []
count = 0
for f in val_files:
    count += 1
    file = f.split('/')[-1].split('\\')[-1]
    val_file_list.append(file)
    if count == 50:
        break;

val_file_list = pd.DataFrame(val_file_list)

val_file_list.to_csv('Dataset/val_half_list.csv', index=False)

