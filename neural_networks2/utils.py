from torchvision.transforms import Resize
from torch import nn



def preprocess_image(image, size=(128, 128), normalize=True):
    resizer = Resize(size)
    image = resizer(image)
    if normalize:
        image = nn.functional.normalize(image)
    return image