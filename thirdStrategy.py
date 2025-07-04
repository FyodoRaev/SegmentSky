import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Используем OpenCV для загрузки изображения
import os
# Импортируем правильные классы
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- 1. Настройка и загрузка модели ---
#
checkpoint_path = "./sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
image_path = "test2.png"  # Путь к вашему изображению


output_dir = "output_segments"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

# Создаем базовую модель SAM2
sam_model = build_sam2(model_cfg_path, checkpoint_path)
sam_model.to(device)


mask_generator = SAM2AutomaticMaskGenerator(sam_model)

print("Модель и генератор масок успешно загружены.")

# --- 2. Загрузка и обработка изображения ---


INPUT_FOLDER = "fold_for_test_cases" # Папка с подготовленными изображениями

# Главная папка для сохранения всех результатов
output_dir = "output_segments"

# Список поддерживаемых расширений файлов изображений
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')




def show_anns(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        m = ann['segmentation']
        mask_color = np.random.random(3)
        img = np.ones((m.shape[0], m.shape[1], 3)) * mask_color
        ax.imshow(np.dstack((img, m * 0.35)))

    plt.axis('off')
    plt.title("Все сгенерированные сегменты", fontsize=16)


# --- 3. Получение списка файлов и запуск цикла ---
print(f"Поиск изображений в папке: '{INPUT_FOLDER}'")
try:
    files_to_process = os.listdir(INPUT_FOLDER)
except FileNotFoundError:
    print(f"[ОШИБКА] Папка '{INPUT_FOLDER}' не найдена. Запустите сначала скрипт предобработки.")
    exit()

# Создаем главную папку для результатов, если ее нет
os.makedirs(output_dir, exist_ok=True)

# Запускаем цикл по каждому файлу
for filename in files_to_process:
    # Проверяем, является ли файл изображением по его расширению
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        print(f"\nПропуск файла '{filename}' (не является изображением).")
        continue

    print(f"\n{'=' * 50}")
    print(f"НАЧАЛО ОБРАБОТКИ: {filename}")
    print(f"{'=' * 50}")

    # --- 3.1. Загрузка и подготовка ОДНОГО изображения ---
    input_path = os.path.join(INPUT_FOLDER, filename)
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        print(f"[ОШИБКА] Не удалось прочитать изображение '{filename}'. Пропускаем.")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Изображение загружено, его размеры: {image_rgb.shape}")

    # --- 3.2. Генерация масок для текущего изображения ---
    print("Начинаю генерацию масок...")
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        masks = mask_generator.generate(image_rgb)
    print(f"Генерация завершена. Найдено {len(masks)} масок.")

    if not masks:
        print("Маски не найдены, переход к следующему файлу.")
        continue

    # --- 3.3. Создание уникальной папки для результатов текущего изображения ---
    image_name_without_ext = os.path.splitext(filename)[0]
    image_output_dir = os.path.join(output_dir, image_name_without_ext)
    raw_mask_dir = os.path.join(image_output_dir, "raw_masks")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(raw_mask_dir, exist_ok=True)
    print(f"Результаты для '{filename}' будут сохранены в: '{image_output_dir}'")

    # --- 3.4. Визуализация и сохранение результатов для текущего изображения ---

    # Сохранение общей визуализации
    show_anns(masks, image_rgb)
    output_path_all = os.path.join(image_output_dir, "all_segments_visualization.png")
    plt.savefig(output_path_all, bbox_inches='tight', dpi=300)
    plt.close()


print(f"\n{'=' * 50}")
print("Все изображения из папки обработаны.")
print(f"{'=' * 50}")