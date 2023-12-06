import json
from torch.utils import data
from PIL import Image


class FashionMnist(data.Dataset):
    def __init__(self, mode, transform=None):
        if mode == 'train':
            with open('data/train.json', 'r') as f:
                self.dataset_info = json.load(f)
        if mode == 'val':
            with open('data/val.json', 'r') as f:
                self.dataset_info = json.load(f)
        self.transform = transform

    def __getitem__(self, index):
        info = self.dataset_info[index]

        imgpath = info['imgpath']
        label = int(info['label'])

        image = Image.open(imgpath)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.dataset_info)



