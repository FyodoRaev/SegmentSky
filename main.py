import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpmath.tests.extratest_gamma import testcases
import os
import glob
import random
# Правильные импорты для SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def show_mask(mask, ax, random_color=False):
    """Отображает маску на изображении."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])  # Голубой цвет для неба
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def main():
    print("Запуск теста сегментации неба с SAM 2...")

    # --- 1. Настройка ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Используется GPU (CUDA).")
    else:
        print("Используется CPU. Это может быть медленно.")

    # Правильные пути для SAM2
    config_file = "configs/sam2.1/sam2.1_hiera_t.yaml"  # Конфигурационный файл
    checkpoint_path = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"

    # Загрузка модели SAM2
    print("Загрузка модели SAM2...")
    try:
        model = build_sam2(config_file, checkpoint_path, device=device)
        predictor = SAM2ImagePredictor(model)
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Убедитесь, что пути к конфигурации и чекпоинту правильные.")
        return

    # --- 2. Загрузка и обработка изображения ---
    import os
    import glob

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

    # Параметры для улучшенного алгоритма
    num_attempts = 5  # Количество попыток поиска неба
    sky_region_height_ratio = 0.3  # Верхние 30% изображения для поиска неба

    def calculate_mask_area(mask):
        """Вычисляет площадь маски (количество пикселей)"""
        return np.sum(mask)

    def generate_sky_points(width, height, num_points, sky_height_ratio=0.3):
        """Генерирует случайные точки в верхней части изображения"""
        sky_height = int(height * sky_height_ratio)
        points = []

        for _ in range(num_points):
            x = random.randint(width // 4, 3 * width // 4)  # Избегаем краев
            y = random.randint(10, sky_height)  # Верхняя часть изображения
            points.append([x, y])

        return np.array(points)

    for i, image_path in enumerate(image_paths):
        print(f"\n--- Обработка изображения {i + 1}/{test_cases}: {os.path.basename(image_path)} ---")

        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError("Изображение не найдено")
            # OpenCV читает в BGR, а модель ожидает RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            print(f"Изображение загружено: {image_rgb.shape}")
        except Exception as e:
            print(f"Ошибка: не удалось загрузить изображение '{image_path}'.")
            print(e)
            continue  # Переходим к следующему изображению вместо завершения программы

        # Устанавливаем изображение для предиктора
        print("Подготовка изображения для модели...")
        predictor.set_image(image_rgb)
        height, width, _ = image_rgb.shape

        # Генерируем случайные точки в верхней части изображения
        candidate_points = generate_sky_points(width, height, num_attempts, sky_region_height_ratio)

        best_mask = None
        best_score = 0
        best_area = 0
        best_point = None
        all_attempts = []

        print(f"Выполняем {num_attempts} попыток поиска неба...")

        for attempt, point in enumerate(candidate_points):
            try:
                # Создаем промпт для текущей точки
                input_point = np.array([point])
                input_label = np.array([1])

                # Получаем маску для текущей точки
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )

                mask = masks[0]
                score = scores[0]
                area = calculate_mask_area(mask)

                # Сохраняем информацию о попытке
                attempt_info = {
                    'point': point,
                    'mask': mask,
                    'score': score,
                    'area': area,
                    'attempt': attempt + 1
                }
                all_attempts.append(attempt_info)

                print(f"  Попытка {attempt + 1}: точка {point}, площадь={area}, score={score:.3f}")

                # Проверяем, лучше ли эта маска
                if area > best_area:
                    best_mask = mask
                    best_score = score
                    best_area = area
                    best_point = point
                    print(f"    ↑ Новая лучшая маска! Площадь: {area}")

            except Exception as e:
                print(f"  Ошибка в попытке {attempt + 1}: {e}")
                continue

        if best_mask is None:
            print("Не удалось получить ни одной маски для этого изображения")
            continue

        print(f"\nЛучший результат: точка {best_point}, площадь={best_area}, score={best_score:.3f}")

        # --- 4. Визуализация лучшего результата ---
        plt.figure(figsize=(15, 10))

        # Основное изображение с лучшей маской
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        show_mask(best_mask, plt.gca())

        # Отображаем все попытки разными цветами
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for j, attempt in enumerate(all_attempts):
            color = colors[j % len(colors)]
            alpha = 1.0 if attempt['point'][0] == best_point[0] and attempt['point'][1] == best_point[1] else 0.5
            size = 300 if attempt['point'][0] == best_point[0] and attempt['point'][1] == best_point[1] else 150
            plt.scatter(attempt['point'][0], attempt['point'][1],
                        color=color, marker='*', s=size,
                        edgecolor='white', linewidth=2, alpha=alpha,
                        label=f"Попытка {attempt['attempt']}: {attempt['area']}")

        plt.title(f"Лучшая сегментация неба\nПлощадь: {best_area}, Score: {best_score:.2f}", fontsize=14)
        plt.axis('off')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Дополнительная визуализация - сравнение всех попыток
        plt.subplot(1, 2, 2)
        areas = [attempt['area'] for attempt in all_attempts]
        scores = [attempt['score'] for attempt in all_attempts]
        attempts_nums = [attempt['attempt'] for attempt in all_attempts]

        plt.bar(attempts_nums, areas, alpha=0.7, color='skyblue', label='Площадь маски')
        plt.xlabel('Номер попытки')
        plt.ylabel('Площадь маски (пиксели)')
        plt.title('Сравнение площадей масок\nпо попыткам')
        plt.grid(True, alpha=0.3)

        # Выделяем лучший результат
        best_attempt_num = next(attempt['attempt'] for attempt in all_attempts
                                if attempt['area'] == best_area)
        plt.bar(best_attempt_num, best_area, color='gold', label='Лучший результат')
        plt.legend()

        # Получаем имя файла без расширения для результата
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        plt.suptitle(f"Анализ сегментации неба: {base_name}", fontsize=16)

        # Сохраняем результат
        result_filename = os.path.join(results_folder, f"sky_segmentation_{base_name}.png")
        plt.savefig(result_filename, dpi=150, bbox_inches='tight')
        print(f"Результат сохранен в '{result_filename}'")

        # Показываем результат
        plt.show()
        plt.close()

    print(f"\nОбработка завершена! Результаты сохранены в папке '{results_folder}'")


if __name__ == "__main__":
    main()