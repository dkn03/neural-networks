import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def __getitem__(self, index):
        vector = self.data[index]
    
        return vector, self.labels[index].to(dtype=torch.int64)
    
    def __len__(self):
        return len(self.data)