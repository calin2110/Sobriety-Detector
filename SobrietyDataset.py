import os.path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SobrietyDataset(Dataset):
    def __init__(self, root_dir, csv_file, expected_size):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.expected_size = expected_size
        self.labels = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[idx,1])
        image_pil = Image.open(img_name)
        image_pil = image_pil.resize(self.expected_size)
        image_arr = np.array(image_pil)
        image_tens = torch.from_numpy(image_arr).permute(2, 0, 1).float()
        image_tens = image_tens.div(255)
        # TODO: https://stats.stackexchange.com/questions/384484/how-should-i-standardize-input-when-fine-tuning-a-cnn
        labels = self.labels.iloc[idx, 2]
        return image_tens, labels
