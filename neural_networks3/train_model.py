import os

import matplotlib.pyplot as plt
import torch
import datetime

from torchvision.io import read_image
from torchvision.transforms import Resize, PILToTensor
from torch.utils.data import random_split, DataLoader, Dataset
from PIL import Image, ImageOps
from torch import nn

from dataset import LesionDataset
from MyUnetModel import MyUnetModel
from utils import training_loop

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def get_images_from_path(path):
    data = []
    for item in os.listdir(path):
        image = Image.open(path + "/" + item + "/" + item + "_Dermoscopic_Image/" + item + '.bmp')
        image = image.resize((128, 128))
        image = PILToTensor()(image)
        image = image.to(device=device, dtype=torch.float32)

        mask = Image.open(path + "/" + item + "/" + item + "_lesion/" + item + '_lesion.bmp')
        mask = mask.resize((128, 128))
        mask = PILToTensor()(mask).squeeze()
        mask = mask.to(dtype=torch.int64, device=device)
        data.append((image, mask))
    return data

train_path = "../nn3_data/PH2 Dataset images/train"
validation_path = "../nn3_data/PH2 Dataset images/train"
train_dataset = LesionDataset(get_images_from_path(train_path))
validation_dataset = LesionDataset(get_images_from_path(validation_path))
train_dataloader, validation_dataloader = DataLoader(train_dataset, batch_size=8), DataLoader(validation_dataset, batch_size=8)

model = MyUnetModel().to(device=device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
model.train()
training_loop(n_epochs=200, optimizer=optimizer, model=model, loss_fn=loss_fn,
              train_loader=train_dataloader, validation_loader=validation_dataloader)

torch.save(model.state_dict(), "my_unet_model2.pt")

model.eval()

result = model(train_dataset[0][0].unsqueeze(0))

print(result.argmax(1).squeeze().shape)

#plt.imshow(train_dataset[0][0].to(dtype=torch.int32).permute(1,2,0), cmap='gray')
plt.imshow(result.argmax(1).squeeze(), cmap='gray')

result = result.squeeze()