import datetime
import os
import torch

from PIL import Image
from torchvision.transforms import PILToTensor

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

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, validation_loader):
    prev_validation_loss = None
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            labels = labels
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # Функция потерь на валидационных данных
        summary_loss = 0
        for img, label in validation_loader:
            img = img
            label = label
            out = model(img)
            summary_loss += loss_fn(out, label)

        validation_loss = summary_loss / len(validation_loader)

        if epoch == 1 or epoch % 5 == 0:

            print('{} Epoch {}, Training loss {}, Validation loss {}'.format(
                datetime.datetime.now(),
                epoch,
                loss_train / len(train_loader),
                validation_loss)
            )
            if prev_validation_loss is not None and prev_validation_loss <= validation_loss:
                print(f"Early stop on epoch {epoch}")
                break
            else:
                prev_validation_loss = validation_loss