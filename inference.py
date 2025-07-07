# src/inference.py
import torch
import numpy as np
import cv2
import argparse
import os
import glob
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import Dinov2ForSkySegmentation


def load_models(dino_weights_path, sam_checkpoint_path, device):
    print("Загрузка DINOv2-based промпт-генератора...")
    model = Dinov2ForSkySegmentation.from_pretrained("facebook/dinov2-base", ignore_mismatched_sizes=True)
    model.classifier.load_state_dict(torch.load(dino_weights_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Загрузка SAM...")
    checkpoint_path = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam = build_sam2(model_cfg_path, checkpoint_path, device=device)

    predictor = SAM2ImagePredictor(sam)

    return model, predictor


def get_sky_prompts(model, image_path, device, dino_image_size=224):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((dino_image_size, dino_image_size), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pixel_values = transform(Image.fromarray(image_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    # Получаем бинарную маску низкого разрешения
    low_res_mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # Находим контуры на маске
    contours, _ = cv2.findContours(low_res_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prompt_points = []
    # Масштабируем координаты промптов до оригинального размера изображения
    h_orig, w_orig = image_rgb.shape[:2]
    h_dino, w_dino = low_res_mask.shape

    for contour in contours:
        # Пропускаем слишком маленькие контуры
        if cv2.contourArea(contour) < 20:
            continue

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Масштабирование
            orig_cX = int(cX * (w_orig / w_dino))
            orig_cY = int(cY * (h_orig / h_dino))
            prompt_points.append([orig_cX, orig_cY])

    return image_rgb, np.array(prompt_points)


def segment_with_sam(predictor, image_rgb, points):
    if len(points) == 0:
        print("Промпт-генератор не нашел кандидатов для неба.")
        return np.zeros(image_rgb.shape[:2], dtype=bool)

    predictor.set_image(image_rgb)
    input_labels = np.ones(len(points), dtype=int)

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=input_labels,
        multimask_output=False,  # Чаще всего одна маска на набор точек работает лучше
    )

    # Объединяем все маски, так как каждая точка могла сгенерировать свою
    final_mask = np.any(masks, axis=0)

    return final_mask


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])  # Dodger blue
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, ax, marker_size=150):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def run_inference(args, dino_given_model, sam_given_predictor):

    dino_model, sam_predictor = dino_given_model, sam_given_predictor

    image_rgb, prompt_points = get_sky_prompts(dino_model, args.image_path, device)

    final_mask = segment_with_sam(sam_predictor, image_rgb, prompt_points)

    # Сохранение и визуализация
    plt.figure(figsize=(15, 10))

    # 1. Оригинал с промптами
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image_rgb)
    if len(prompt_points) > 0:
        show_points(prompt_points, ax1)
    ax1.set_title("Оригинал + Промпты")
    ax1.axis('off')

    # 2. Результат сегментации
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(image_rgb)
    show_mask(final_mask, ax2)
    ax2.set_title("Маска неба (SAM)")
    ax2.axis('off')

    # 3. Только маска
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(final_mask, cmap='gray')
    ax3.set_title("Бинарная маска")
    ax3.axis('off')

    plt.tight_layout()

    basename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_path_viz = os.path.join(args.output_dir, f"{basename}_sky_segmentation.png")
    output_path_mask = os.path.join(args.output_dir, f"{basename}_sky_mask.png")

    plt.savefig(output_path_viz)
    cv2.imwrite(output_path_mask, final_mask.astype(np.uint8) * 255)

    print(f"Визуализация сохранена в: {output_path_viz}")
    print(f"Бинарная маска сохранена в: {output_path_mask}")


if __name__ == "__main__":

    # Путь к папке с тестовыми изображениями
    test_cases_dir = "../test_cases"

    # Получаем список всех изображений в папке
    # Поддерживаем разные форматы изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_cases_dir, ext)))

    if not image_paths:
        print(f"Не найдено изображений в папке {test_cases_dir}")
        exit(1)

    print(f"Найдено {len(image_paths)} изображений для обработки\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_weights = "../checkpoints/sky_detector_head_epoch_3.pth"
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    dino_to_inference, sam_to_inference = load_models(dino_weights, sam_checkpoint, device)
    print(f"  Веса DINO:   {dino_weights}")
    print(f"  Веса SAM:    {sam_checkpoint}\n")

    # Обрабатываем каждое изображение
    for idx, image_path in enumerate(image_paths, 1):
        print(f"Обработка изображения {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
        print("-" * 50)

        args = argparse.Namespace(
            image_path=image_path,

            # 2. Путь к весам обученной "головы" для DINOv2.
            #    Убедитесь, что имя файла совпадает с тем, что сгенерировал train.py


            # 4. Тип модели SAM (должен соответствовать файлу весов: vit_h, vit_l, vit_b)
            sam_model="vit_h",

            # 5. Директория, куда будут сохранены результаты.
            output_dir="../output"
            # ---------------------------
        )

        # Убедимся, что директория для вывода существует
        os.makedirs(args.output_dir, exist_ok=True)

        # Запускаем основную функцию
        print(f"  Изображение: {args.image_path}")


        run_inference(args, dino_to_inference, sam_to_inference)

        print(f"\nИнференс для {os.path.basename(image_path)} завершен.\n")

    print(f"\nВсе изображения обработаны. Результаты сохранены в {args.output_dir}")