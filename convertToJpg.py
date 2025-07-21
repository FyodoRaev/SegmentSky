import os
from PIL import Image
import glob


def convert_images_to_jpg(input_folder, supported_formats=None):
    """
    Конвертирует изображения из различных форматов в .jpg

    Args:
        input_folder: путь к папке с изображениями
        supported_formats: список поддерживаемых форматов для конвертации
    """
    if supported_formats is None:
        supported_formats = ['.png', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif']

    # Получаем все файлы в папке
    all_files = os.listdir(input_folder)

    converted_count = 0
    skipped_count = 0

    for filename in all_files:
        file_path = os.path.join(input_folder, filename)

        # Проверяем, что это файл (не папка)
        if not os.path.isfile(file_path):
            continue

        # Получаем расширение файла
        name, ext = os.path.splitext(filename)

        # Если уже .jpg, пропускаем
        if ext == '.jpg':
            skipped_count += 1
            continue

        # Если формат поддерживается для конвертации
        # if ext_lower in supported_formats:
        if True:
                # Открываем изображение
                with Image.open(file_path) as img:
                    # Конвертируем в RGB если необходимо (для JPEG)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Создаем белый фон для прозрачных изображений
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Создаем новое имя файла с расширением .jpg
                    new_filename = name + '.jpg'
                    new_file_path = os.path.join(input_folder, new_filename)

                    # Сохраняем в формате JPEG с высоким качеством
                    img.save(new_file_path, 'JPEG', quality=95, optimize=True)

                    print(f"Конвертировано: {filename} -> {new_filename}")
                    converted_count += 1

                    # Удаляем оригинальный файл
                    os.remove(file_path)

    print(f"\nГотово!")
    print(f"Конвертировано файлов: {converted_count}")
    print(f"Пропущено .jpg файлов: {skipped_count}")


if __name__ == "__main__":
    # Путь к папке с изображениями
    train_images_folder = "train_data/train_images"

    # Проверяем существование папки
    if not os.path.exists(train_images_folder):
        print(f"Папка {train_images_folder} не найдена!")
        exit(1)

    print(f"Начинаем конвертацию изображений в папке: {train_images_folder}")
    print("Поддерживаемые форматы для конвертации: .png, .jpeg, .bmp, .tiff, .tif, .webp, .gif")

    # Запускаем конвертацию
    convert_images_to_jpg(train_images_folder)