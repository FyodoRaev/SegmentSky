# src/dataset.py
from asyncio import timeout

import torch
from datasets import load_dataset, DownloadConfig
import aiohttp
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


DATASET_NAME = "zhoubolei/scene_parse_150"

class SkyDataset(Dataset):
    def __init__(self, split='train', image_size=224, cache_dir="../data"):
        super().__init__()
        self.image_size = image_size

        # Увеличиваем таймаут до 300 секунд (5 минут)
        # Можете поставить и больше, если нужно

        print(f"Загрузка датасета ADE20K для сплита '{split}'...")
        # scene_parse_150 - это официальное название ADE20K в Hugging Face Datasets
        self.ds = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})

        # Находим ID класса 'sky'. В ADE20K это 3.
        self.sky_label_id = 3

        print("Фильтрация датасета: оставляем только изображения с небом...")
        self.filtered_indices = self._filter_dataset()
        print(f"Найдено {len(self.filtered_indices)} изображений с небом.")

        # Трансформации для изображения
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Трансформации для маски
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _filter_dataset(self):
        indices = []
        for i in range(len(self.ds)):
            # Загружаем маску как numpy-массив
            mask = np.array(self.ds[i]['annotation'])
            # Проверяем, есть ли в маске пиксели с классом 'sky'
            if np.any(mask == self.sky_label_id):
                indices.append(i)
        return indices

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        item = self.ds[original_idx]

        image = item['image'].convert("RGB")
        annotation = item['annotation']

        # Создаем бинарную маску: 1 - небо, 0 - все остальное
        mask_np = np.array(annotation)
        binary_mask_np = (mask_np == self.sky_label_id).astype(np.uint8) * 255
        binary_mask = Image.fromarray(binary_mask_np)

        pixel_values = self.image_transform(image)
        labels = self.mask_transform(binary_mask)

        return {"pixel_values": pixel_values, "labels": labels}