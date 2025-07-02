import os
import numpy as np
from datasets import load_dataset
from PIL import Image

# --- Параметры ---
DATASET_NAME = "zhoubolei/scene_parse_150"
BASE_FOLDER = "fold_for_test_cases"
IMAGES_FOLDER = os.path.join(BASE_FOLDER, "images")
GT_MASKS_FOLDER = os.path.join(BASE_FOLDER, "masks_ground_truth")
NUM_IMAGES = 30
SPLIT = "validation"

# ID класса "небо" в датасете ADE20K (на котором основан scene_parse_150)
SKY_CLASS_ID = 3


def download_and_prepare_dataset():
    """
    Загружает изображения и их эталонные маски, сохраняя их в отдельные папки.
    """
    print(f"Загрузка информации о датасете '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    # Создаем папки
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    os.makedirs(GT_MASKS_FOLDER, exist_ok=True)
    print(f"Изображения будут сохранены в: '{IMAGES_FOLDER}'")
    print(f"Эталонные маски (Ground Truth) будут сохранены в: '{GT_MASKS_FOLDER}'")

    subset = dataset.take(NUM_IMAGES)
    print(f"\nНачинается загрузка и сохранение {NUM_IMAGES} пар (изображение + маска)...")

    count = 0
    for i, example in enumerate(subset):
        try:
            # --- 1. Обработка и сохранение изображения ---
            image = example['image'].convert("RGB")
            base_filename = f"scene_{str(i + 1).zfill(3)}"
            image_filename = f"{base_filename}.jpg"
            image_filepath = os.path.join(IMAGES_FOLDER, image_filename)
            image.save(image_filepath, "JPEG")

            # --- 2. Обработка и сохранение эталонной маски неба ---
            # 'annotation' - это маска со всеми классами (0, 1, 2, ... 150)
            annotation_image = example['annotation']
            annotation_np = np.array(annotation_image)

            # Создаем бинарную маску: 255 где небо (id=2), и 0 где все остальное
            sky_gt_mask_np = (annotation_np == SKY_CLASS_ID).astype(np.uint8) * 255

            # Конвертируем обратно в изображение PIL и сохраняем
            sky_gt_mask_image = Image.fromarray(sky_gt_mask_np)
            mask_filename = f"{base_filename}_mask.png"
            mask_filepath = os.path.join(GT_MASKS_FOLDER, mask_filename)
            # PNG - лучший формат для масок, т.к. сжатие без потерь
            sky_gt_mask_image.save(mask_filepath, "PNG")

            count += 1
            if (i + 1) % 20 == 0:
                print(f"  ...сохранено {i + 1}/{NUM_IMAGES} пар")

        except Exception as e:
            print(f"Не удалось обработать пару {i + 1}: {e}")

    print(f"\nГотово! Успешно сохранено {count} из {NUM_IMAGES} пар.")
    print("Теперь вы можете запустить ваш основной скрипт для тестирования.")


if __name__ == "__main__":
    download_and_prepare_dataset()