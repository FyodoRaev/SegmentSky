import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2
import os
import glob


class SkySegmenter:
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-ade-640-640"):
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # ADE20K класс для неба
        self.sky_class_id = 2  # В ADE20K небо имеет id=2

    def segment_sky(self, image_path):
        # Загрузка изображения
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # Предобработка
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Инференс
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Постобработка
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # height, width
            mode="bilinear",
            align_corners=False
        )

        # Получение предсказаний
        pred_seg = upsampled_logits.argmax(dim=1).cpu().numpy()[0]

        # Создание бинарной маски для неба
        sky_mask = (pred_seg == self.sky_class_id).astype(np.uint8) * 255

        return sky_mask, pred_seg

    def visualize_result(self, image_path, sky_mask):
        # Загрузка оригинального изображения
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Создание визуализации
        overlay = original.copy()
        overlay[sky_mask > 0] = [135, 206, 235]  # Небесно-голубой цвет

        # Смешивание с оригиналом
        result = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

        return result


def process_dataset():
    # --- 2. Загрузка и обработка изображения ---
    test_folder = "fold_for_test_cases"  # папка с тестовыми изображениями

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

    # Получаем список всех изображений в папке
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_folder, extension)))
        image_paths.extend(glob.glob(os.path.join(test_folder, extension.upper())))

    if not image_paths:
        print(f"Ошибка: в папке '{test_folder}' не найдено изображений.")
        print("Убедитесь, что папка существует и содержит изображения в форматах: jpg, jpeg, png, bmp, tiff")
        return

    test_cases = len(image_paths)
    print(f"Найдено {test_cases} изображений для обработки")

    # Создаем папку для результатов, если её нет
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)

    # Инициализация сегментатора
    print("Инициализация модели...")
    segmenter = SkySegmenter()

    # Обработка каждого изображения
    for i, image_path in enumerate(image_paths):
        print(f"\n--- Обработка изображения {i + 1}/{test_cases}: {os.path.basename(image_path)} ---")

        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError("Изображение не найдено")

            # Сегментация
            sky_mask, _ = segmenter.segment_sky(image_path)

            # Визуализация
            visualization = segmenter.visualize_result(image_path, sky_mask)

            # Сохранение результатов
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # Создаем папку для этого изображения
            image_results_folder = os.path.join(results_folder, base_name)
            os.makedirs(image_results_folder, exist_ok=True)

            # Сохраняем маску
            mask_path = os.path.join(image_results_folder, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, sky_mask)

            # Сохраняем визуализацию
            vis_path = os.path.join(image_results_folder, f"{base_name}_visualization.png")
            cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            print(f"  ✓ Результаты сохранены в: {image_results_folder}")

        except Exception as e:
            print(f"  ✗ Ошибка при обработке: {str(e)}")


# Запуск
if __name__ == "__main__":
    process_dataset()