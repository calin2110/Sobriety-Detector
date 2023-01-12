import pandas as pd
import numpy as np
import os
import shutil

data = pd.read_csv("data/data_info.csv")

#Part that merges the sober and drunk labels
drunk = data[data['Label'] == 1].copy()
sober = data[data['Label'] == 0].copy()
sober_min_idx = sober['Name'].apply(lambda x: int(x.split('.')[0].split('_')[2])).min()
drunk_min_idx = drunk['Name'].apply(lambda x: int(x.split('.')[0].split('_')[2])).min()
drunk['Ord'] = drunk['Name'].apply(lambda x: (int(x.split('.')[0].split('_')[2]) - drunk_min_idx)//8)
sober['Ord'] = sober['Name'].apply(lambda x: (int(x.split('.')[0].split('_')[2]) - sober_min_idx)//5)
data = pd.concat([drunk, sober], ignore_index=True)
data = data.sort_values(by='Ord')
data = data.drop(columns=['Ord'])

train_folder = "data/train/"
test_folder = "data/test/"
val_folder = "data/validation/"

train_split = 0.6
test_split = 0.2
val_split = 0.2

train_data, test_data, val_data = np.split(data, [int(train_split * len(data)), int((train_split+test_split) * len(data))])

train_data.to_csv(os.path.join(train_folder, "train.csv"), index=False)
test_data.to_csv(os.path.join(test_folder, "test.csv"), index=False)
val_data.to_csv(os.path.join(val_folder, "validation.csv"), index=False)

for image in train_data['Name']:
    shutil.copy("data/" + image, os.path.join(train_folder, image))
for image in test_data['Name']:
    shutil.copy("data/" + image, os.path.join(test_folder, image))
for image in val_data['Name']:
    shutil.copy("data/" + image, os.path.join(val_folder, image))
