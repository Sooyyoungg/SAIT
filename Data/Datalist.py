
import glob
import pandas as pd

# make data list
file_dir = './Data/Test/SEM/*'
files = sorted(glob.glob(file_dir))

file_list = []
for f in files:
    file = f.split('/')[-1].split('\\')[-1]
    file_list.append(file)
print(len(file_list))
print(file_list[0])

file_list = pd.DataFrame(file_list)
print(file_list)

file_list.to_csv('Data/test_list.csv', index=False)

# with open('Data/test_list.csv', 'w', newline='') as f:
#     write = csv.writer(f)
#     write.writerow(file_list)