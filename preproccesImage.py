import cv2
import os

# --- 1. Настройка ---
# Папки для исходных и обработанных изображений
INPUT_FOLDER = 'raw_images'
OUTPUT_FOLDER = 'fold_for_test_cases'

# Максимальный размер по ширине или высоте
MAX_DIMENSION = 2000

# Список поддерживаемых расширений файлов изображений
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# --- 2. Подготовка ---
# Создаем папки, если они не существуют
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Исходные изображения будут взяты из: '{INPUT_FOLDER}'")
print(f"Обработанные изображения будут сохранены в: '{OUTPUT_FOLDER}'")
print(f"Максимальный размер изображения: {MAX_DIMENSION}x{MAX_DIMENSION} пикселей\n")

# --- 3. Обработка изображений ---
# Получаем список файлов в исходной папке
try:
    files_to_process = os.listdir(INPUT_FOLDER)
except FileNotFoundError:
    print(f"[ОШИБКА] Папка '{INPUT_FOLDER}' не найдена. Пожалуйста, создайте ее.")
    exit()

if not files_to_process:
    print(f"Папка '{INPUT_FOLDER}' пуста. Добавьте изображения для обработки.")
    exit()

processed_count = 0
copied_count = 0

for filename in files_to_process:
    # Проверяем, является ли файл изображением по его расширению
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        print(f"Пропуск файла '{filename}' (не является изображением).")
        continue

    # Составляем полный путь к файлу
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    # Читаем изображение
    image = cv2.imread(input_path)
    if image is None:
        print(f"Не удалось прочитать изображение '{filename}'. Пропускаем.")
        continue

    # Получаем его текущие размеры
    h, w = image.shape[:2]

    # Проверяем, нужно ли изменять размер
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        print(f"Обработка '{filename}'... (исходный размер: {w}x{h})")

        # Вычисляем коэффициент масштабирования, сохраняя пропорции
        # Мы уменьшаем изображение по его большей стороне
        if h > w:
            ratio = MAX_DIMENSION / h
            new_h = MAX_DIMENSION
            new_w = int(w * ratio)
        else:
            ratio = MAX_DIMENSION / w
            new_w = MAX_DIMENSION
            new_h = int(h * ratio)

        # Изменяем размер изображения с помощью интерполяции cv2.INTER_AREA
        # Этот метод лучше всего подходит для уменьшения изображений,
        # так как он предотвращает появление артефактов ("муара").
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Сохраняем измененное изображение
        cv2.imwrite(output_path, resized_image)
        print(f"  -> Изображение уменьшено до {new_w}x{new_h} и сохранено в '{output_path}'")
        processed_count += 1
    else:
        # Если изображение уже подходит по размеру, просто копируем его
        print(f"Копирование '{filename}'... (размер {w}x{h} уже в пределах нормы)")
        cv2.imwrite(output_path, image)
        print(f"  -> Изображение скопировано в '{output_path}'")
        copied_count += 1

print(f"\nОбработка завершена.")
print(f"Уменьшено и сохранено: {processed_count} изображений.")
print(f"Скопировано без изменений: {copied_count} изображений.")
