from torch.utils.data import Dataset


class LesionDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        image = self.data[index][0]
        mask = self.data[index][1]
        return image, mask

    def __len__(self):
        return len(self.data)
