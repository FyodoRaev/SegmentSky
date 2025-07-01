import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
from datetime import datetime

# Правильные импорты для SAM2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def save_segment_info(segment_data, save_path):
    """Сохраняет информацию о сегменте в JSON файл"""
    # Конвертируем numpy arrays в списки для JSON
    segment_info = {
        'area': int(segment_data['area']),
        'bbox': segment_data['bbox'],
        'predicted_iou': float(segment_data['predicted_iou']),
        'point_coords': segment_data['point_coords'],
        'stability_score': float(segment_data['stability_score']),
        'crop_box': segment_data['crop_box']
    }

    with open(save_path, 'w') as f:
        json.dump(segment_info, f, indent=2)


def save_segment_mask(mask, save_path):
    """Сохраняет маску сегмента как изображение"""
    # Конвертируем boolean маску в uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_uint8)


def extract_segment_crop(image, mask, bbox, padding=10):
    """Извлекает область изображения, соответствующую сегменту"""
    x, y, w, h = bbox

    # Добавляем отступы
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(image.shape[1], int(x + w + padding))
    y2 = min(image.shape[0], int(y + h + padding))

    # Извлекаем область
    crop = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    # Применяем маску (делаем фон прозрачным/черным)
    if len(crop.shape) == 3:
        # Создаем RGBA изображение
        crop_rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
        crop_rgba[:, :, :3] = crop
        crop_rgba[:, :, 3] = crop_mask * 255  # Альфа канал
        return crop_rgba
    else:
        return crop * crop_mask


def visualize_all_segments(image, segments, max_display=20):
    """Создает визуализацию всех найденных сегментов"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    # Показываем оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Оригинал')
    axes[0].axis('off')

    # Показываем первые max_display-1 сегментов
    for i in range(min(len(segments), max_display - 1)):
        segment = segments[i]
        mask = segment['segmentation']

        # Создаем цветную маску
        colored_mask = np.zeros((*mask.shape, 3))
        color = np.random.rand(3)
        colored_mask[mask] = color

        axes[i + 1].imshow(image)
        axes[i + 1].imshow(colored_mask, alpha=0.6)
        axes[i + 1].set_title(f'Сегмент {i + 1}\nПлощадь: {segment["area"]}')
        axes[i + 1].axis('off')

    # Скрываем неиспользуемые subplot'ы
    for i in range(len(segments) + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def main():
    print("Запуск извлечения всех сегментов с SAM 2...")

    # --- 1. Настройка модели ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    # Загрузка модели SAM2
    config_file = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint_path = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"

    print("Загрузка модели SAM2...")
    try:
        model = build_sam2(config_file, checkpoint_path, device=device)

        # Создаем автоматический генератор масок
        mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=32,  # Количество точек на сторону для сетки
            pred_iou_thresh=0.7,  # Порог IoU для фильтрации
            stability_score_thresh=0.8,  # Порог стабильности
            crop_n_layers=1,  # Количество слоев обрезки
            crop_n_points_downscale_factor=2,  # Фактор уменьшения точек
            min_mask_region_area=500,  # Минимальная площадь сегмента
        )
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # --- 2. Настройка папок ---
    test_folder = "fold_for_test_cases"
    segments_folder = "extracted_segments"

    # Создаем основную папку для сегментов
    os.makedirs(segments_folder, exist_ok=True)

    # Поиск изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_folder, extension)))
        image_paths.extend(glob.glob(os.path.join(test_folder, extension.upper())))

    if not image_paths:
        print(f"Ошибка: в папке '{test_folder}' не найдено изображений.")
        return

    print(f"Найдено {len(image_paths)} изображений для обработки")

    # --- 3. Обработка каждого изображения ---
    num_runs = 1  # Количество запусков для каждого изображения

    for img_idx, image_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n--- Обработка изображения {img_idx + 1}/{len(image_paths)}: {base_name} ---")

        # Загрузка изображения
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError("Изображение не найдено")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            print(f"Изображение загружено: {image_rgb.shape}")
        except Exception as e:
            print(f"Ошибка загрузки {image_path}: {e}")
            continue

        # Создаем папку для этого изображения
        image_segments_folder = os.path.join(segments_folder, base_name)
        os.makedirs(image_segments_folder, exist_ok=True)

        all_segments = []  # Собираем все сегменты со всех запусков

        # Запускаем сегментацию несколько раз
        for run_idx in range(num_runs):
            print(f"  Запуск {run_idx + 1}/{num_runs}...")

            try:
                # Генерируем маски
                segments = mask_generator.generate(image_rgb)
                print(f"    Найдено {len(segments)} сегментов")

                # Создаем папку для этого запуска
                run_folder = os.path.join(image_segments_folder, f"run_{run_idx + 1}")
                os.makedirs(run_folder, exist_ok=True)

                # Сохраняем каждый сегмент
                for seg_idx, segment in enumerate(segments):
                    segment_folder = os.path.join(run_folder, f"segment_{seg_idx:03d}")
                    os.makedirs(segment_folder, exist_ok=True)

                    # Сохраняем маску
                    mask_path = os.path.join(segment_folder, "mask.png")
                    save_segment_mask(segment['segmentation'], mask_path)

                    # Сохраняем информацию о сегменте
                    info_path = os.path.join(segment_folder, "info.json")
                    save_segment_info(segment, info_path)

                    # Извлекаем и сохраняем область сегмента
                    crop = extract_segment_crop(image_rgb, segment['segmentation'], segment['bbox'])
                    crop_path = os.path.join(segment_folder, "crop.png")
                    if crop.shape[2] == 4:  # RGBA
                        # Конвертируем в BGR для OpenCV
                        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGRA)
                        cv2.imwrite(crop_path, crop_bgr)
                    else:
                        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(crop_path, crop_bgr)

                # Добавляем сегменты к общему списку
                for segment in segments:
                    segment['run'] = run_idx + 1
                all_segments.extend(segments)

                # Создаем визуализацию для этого запуска
                if len(segments) > 0:
                    fig = visualize_all_segments(image_rgb, segments)
                    viz_path = os.path.join(run_folder, "visualization.png")
                    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

            except Exception as e:
                print(f"    Ошибка в запуске {run_idx + 1}: {e}")
                continue

        # Сохраняем сводную информацию
        summary = {
            'image_name': base_name,
            'image_shape': image_rgb.shape,
            'total_runs': num_runs,
            'total_segments': len(all_segments),
            'segments_per_run': [len([s for s in all_segments if s['run'] == i]) for i in range(1, num_runs + 1)],
            'processing_date': datetime.now().isoformat()
        }

        summary_path = os.path.join(image_segments_folder, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  Всего извлечено {len(all_segments)} сегментов")
        print(f"  Результаты сохранены в: {image_segments_folder}")

    print(f"\nИзвлечение сегментов завершено!")
    print(f"Все результаты сохранены в папке: {segments_folder}")
    print("\nСтруктура папок:")
    print("extracted_segments/")
    print("  ├── image1/")
    print("  │   ├── run_1/")
    print("  │   │   ├── segment_001/")
    print("  │   │   │   ├── mask.png")
    print("  │   │   │   ├── crop.png")
    print("  │   │   │   └── info.json")
    print("  │   │   └── visualization.png")
    print("  │   ├── run_2/")
    print("  │   └── summary.json")


if __name__ == "__main__":
    main()