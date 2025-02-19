import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any
from PIL import Image
import clip
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=0)

class Images(Dataset):
    def __init__(self, image_list, transform) -> None:
        super().__init__()
        self._image_list = image_list
        self._transform = transform

    def __len__(self):
        return len(self._image_list)
    
    def __getitem__(self, index) -> Any:
        image_path = self._image_list[index]
        image = Image.open(image_path)
        image = self._transform(image)
        return {
            "image": image,
            "img_path": image_path
        }
