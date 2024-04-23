import datetime
import os
import torch

from PIL import Image
from torchvision.transforms import PILToTensor
from numpy import genfromtxt

def get_images_from_path(path, device):
    data = []
    for item in os.listdir(path):
        image = Image.open(path + "/" + item + "/" + item + "_Dermoscopic_Image/" + item + '.bmp')
        image = image.resize((256, 256))
        image = PILToTensor()(image)
        image = image.to(device=device, dtype=torch.float32)

        mask = Image.open(path + "/" + item + "/" + item + "_lesion/" + item + '_lesion.bmp')
        mask = mask.resize((256, 256))
        mask = PILToTensor()(mask).squeeze()
        mask = mask.to(dtype=torch.int64, device=device)
        data.append((image, mask))
    return data

def read_data(data_path):
    files = os.listdir(data_path)
    file_path = data_path + files[0]
    train_data = torch.tensor(genfromtxt(file_path)).unsqueeze(2)
    for file in os.listdir(data_path)[1:]:
        file_path = data_path + file
        new_vector = torch.tensor(genfromtxt(file_path)).unsqueeze(2)
        train_data = torch.concat((train_data, new_vector), 2)

    return train_data.to(dtype=torch.float32)