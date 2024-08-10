import os
import torch
import nibabel as nib
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root 
        self.mode = mode
        self.data_path = os.path.join(self.root,self.mode)
        self.data = sorted(os.listdir(self.data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_name = self.data[index]
        label = torch.tensor([int(data_name.split("_")[0])])
        data = nib.load(os.path.join(self.data_path,data_name)).get_fdata()
        return torch.tensor(data).unsqueeze(0),label
