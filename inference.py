import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model.u2net import U2NET  # путь может быть 'models/U2NET' в вашем проекте

# 1. Параметры
model_path = 'saved_models/u2net_bce_itr_1900_train_0.376974_tar_0.043999.pth'  # ваш файл весов
test_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')  # папка с новыми изображениями
output_dir = os.path.join(os.getcwd(), 'u2net_results')
os.makedirs(output_dir, exist_ok=True)

# 1.1. Параметры для визуализации
# Цвет для наложения маски (в формате BGR: Синий, Зеленый, Красный)
OVERLAY_COLOR_BGR = [0, 0, 255]  # Красный
# Прозрачность наложения (0.4 - это 40% цвета маски и 60% исходного изображения)
ALPHA = 0.4
# Стиль тепловой карты (другие варианты: cv2.COLORMAP_HOT, cv2.COLORMAP_INFERNO, etc.)
COLORMAP_STYLE = cv2.COLORMAP_JET

# 2. Загружаем модель
print("Загрузка модели...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = U2NET(3, 1)
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()
print(f"Модель загружена на {device}.")

# 3. Трансформации
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Инференс и создание визуализаций
print(f"Обработка изображений из папки: {test_dir}")
for img_path in glob.glob(os.path.join(test_dir, '*')):
    # --- Инференс модели (без изменений) ---
    img_pil = Image.open(img_path).convert('RGB')
    w, h = img_pil.size
    inp = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *rest = net(inp)
        # Получаем карту вероятностей (значения от 0.0 до 1.0)
        mask_prob = torch.sigmoid(d1).squeeze().cpu().numpy()

    # Масштабируем карту вероятностей до исходного размера изображения
    mask_resized = cv2.resize(mask_prob, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- Блок создания визуализаций ---

    # 1. Исходное изображение (BGR для OpenCV)
    original_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # --- Создание комбинированного изображения (оригинал, наложение, ч/б маска) ---

    # Бинарная маска для наложения и ч/б визуализации
    bin_mask = (mask_resized > 0.57).astype(np.uint8)

    # Изображение с полупрозрачной маской
    overlay_solid = original_bgr.copy()
    overlay_solid[bin_mask > 0] = OVERLAY_COLOR_BGR
    overlayed_image = cv2.addWeighted(overlay_solid, ALPHA, original_bgr, 1 - ALPHA, 0)

    # Черно-белая маска (3-х канальная для объединения)
    bw_mask_visual = bin_mask * 255
    bw_mask_bgr = cv2.cvtColor(bw_mask_visual, cv2.COLOR_GRAY2BGR)

    # Объединяем три изображения в одно
    combined_image = cv2.hconcat([original_bgr, overlayed_image, bw_mask_bgr])

    # --- Создание тепловой карты (Heatmap) ---

    # Конвертируем вероятности (float 0.0-1.0) в 8-битный формат (int 0-255)
    heatmap_gray = (mask_resized * 255).astype(np.uint8)
    # Применяем цветовую карту для создания цветной тепловой карты
    heatmap_color = cv2.applyColorMap(heatmap_gray, COLORMAP_STYLE)

    # --- Сохранение результатов ---

    name = os.path.splitext(os.path.basename(img_path))[0]

    # Сохраняем комбинированное изображение
    output_path_combined = os.path.join(output_dir, f'{name}_result.png')
    cv2.imwrite(output_path_combined, combined_image)
    print(f'Сохранен результат для {name} в {output_path_combined}')

    # Сохраняем тепловую карту как отдельный файл
    output_path_heatmap = os.path.join(output_dir, f'{name}_heatmap.png')
    cv2.imwrite(output_path_heatmap, heatmap_color)
    print(f'Сохранена тепловая карта для {name} в {output_path_heatmap}')

print("Готово!")