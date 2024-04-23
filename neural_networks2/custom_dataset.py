from torch.utils.data import Dataset
from torchvision.transforms import Resize
import torch.nn as nn

class FoodDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        image = self.data[index][1]
        label = self.data[index][0]
        return image, label

    def __len__(self):
        return len(self.data)