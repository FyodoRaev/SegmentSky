import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from U2NET import U2NET  # путь может быть 'models/U2NET' в вашем проекте

# 1. Параметры
model_path = 'saved_models/u2net_sky.pth'  # ваш файл весов
test_dir = os.path.join(os.getcwd(), 'dataset', 'test_images')  # папка с новыми изображениями
output_dir = os.path.join(os.getcwd(), 'u2net_results')
os.makedirs(output_dir, exist_ok=True)

# 2. Загружаем модель
net = U2NET(3, 1)  # U2NET или U2NETP
net.load_state_dict(torch.load(model_path, map_location='cpu'))
net.eval()

# 3. Трансформации
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Инференс
for img_path in glob.glob(os.path.join(test_dir, '*')):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    inp = transform(img).unsqueeze(0)

    with torch.no_grad():
        d1, *rest = net(inp)
        mask = torch.sigmoid(d1[:, 0, :, :]).squeeze().cpu().numpy()

    # восстанавливаем исходный размер
    mask = cv2.resize(mask, (w, h))
    bin_mask = (mask > 0.5).astype(np.uint8) * 255

    name = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(output_dir, f'{name}.png'), bin_mask)
    print(f'Saved mask for {name}')