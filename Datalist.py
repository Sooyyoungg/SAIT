
import glob
import pandas as pd

# make data list
train_file_dir = './Dataset/Train/SEM/*'
val_file_dir = './Dataset/Validation/SEM/*'
test_file_dir = './Dataset/Test/SEM/*'
train_files = sorted(glob.glob(train_file_dir))
val_files = sorted(glob.glob(val_file_dir))
test_files = sorted(glob.glob(test_file_dir))

train_file_list = []
for f in train_files:
    file = f.split('/')[-1].split('\\')[-1]
    train_file_list.append(file)
print(len(train_file_list))
print(train_file_list[0])

val_file_list = []
for f in train_files:
    file = f.split('/')[-1].split('\\')[-1]
    val_file_list.append(file)

test_file_list = []
for f in train_files:
    file = f.split('/')[-1].split('\\')[-1]
    test_file_list.append(file)

train_file_list = pd.DataFrame(train_file_list)
val_file_list = pd.DataFrame(val_file_list)
test_file_list = pd.DataFrame(test_file_list)

train_file_list.to_csv('Dataset/train_list.csv', index=False)
val_file_list.to_csv('Dataset/val_list.csv', index=False)
test_file_list.to_csv('Dataset/test_list.csv', index=False)

