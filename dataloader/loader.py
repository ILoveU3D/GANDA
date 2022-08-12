import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from options import root

class CrezhFront(Dataset):
    def __init__(self):
        self.items = os.listdir(root)

    def __getitem__(self,index):
        img = Image.open(os.path.join(root, self.items[index]))
        return transforms.ToTensor()(img)

    def __len__(self):
        return len(self.items)