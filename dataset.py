# src/dataset.py
import torch
from datasets import load_dataset
import aiohttp
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random  # Добавляем импорт для случайных операций

DATASET_NAME = "zhoubolei/scene_parse_150"


class SkyDataset(Dataset):
    def __init__(self, split='train', image_size=500, cache_dir="../data"):
        super().__init__()
        self.image_size = image_size


        print(f"Загрузка датасета ADE20K для сплита '{split}'...")
        # Увеличиваем таймаут, чтобы избежать ошибок при скачивании большого датасета
        self.ds = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir,
                               storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})

        # ID класса 'sky' в ADE20K - это 3.
        self.sky_label_id = 3

        print("Подготовка и сэмплирование датасета...")
        self.final_indices, self.sky_indices_set = self._prepare_indices()

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

    def _prepare_indices(self):
        """
        Разделяет датасет на изображения с небом и без, затем формирует сбалансированную выборку.
        """
        sky_indices = []
        no_sky_indices = []

        print("Анализ датасета (это может занять некоторое время)...")
        for i in range(len(self.ds)):
            # Загружаем маску как numpy-массив
            mask = np.array(self.ds[i]['annotation'])
            # Проверяем, есть ли в маске пиксели с классом 'sky'
            if np.any(mask == self.sky_label_id):
                sky_indices.append(i)
            else:
                no_sky_indices.append(i)

        print(f"Найдено {len(sky_indices)} изображений с небом и {len(no_sky_indices)} без неба.")

        # Берем столько же изображений без неба, сколько есть с небом (или меньше, если их не хватает)
        num_no_sky_to_sample = min(len(sky_indices), len(no_sky_indices))
        sampled_no_sky_indices = random.sample(no_sky_indices, num_no_sky_to_sample)

        print(
            f"В итоговую выборку добавлено {len(sky_indices)} изображений с небом и {len(sampled_no_sky_indices)} изображений без неба.")

        # Объединяем и перемешиваем индексы
        final_indices = sky_indices + sampled_no_sky_indices
        random.shuffle(final_indices)

        # Сохраняем set индексов с небом для быстрой проверки в __getitem__
        sky_indices_set = set(sky_indices)

        return final_indices, sky_indices_set

    def __len__(self):
        return len(self.final_indices)

    def __getitem__(self, idx):
        # Получаем оригинальный индекс из нашего перемешанного списка
        original_idx = self.final_indices[idx]
        item = self.ds[original_idx]

        image = item['image'].convert("RGB")
        annotation = item['annotation']

        # Создаем бинарную маску: 1 - небо, 0 - все остальное
        # Эта логика работает и для изображений без неба (маска будет полностью нулевой)
        mask_np = np.array(annotation)
        binary_mask_np = (mask_np == self.sky_label_id).astype(np.uint8) * 255
        binary_mask = Image.fromarray(binary_mask_np)

        # Применяем трансформации
        pixel_values = self.image_transform(image)
        labels = self.mask_transform(binary_mask)

        # Для изображений без неба labels будет тензором из нулей, что и требуется
        return {"pixel_values": pixel_values, "labels": labels}