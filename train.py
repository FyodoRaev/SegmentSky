# src/train.py
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import Dinov2Config
from torch.optim import AdamW
from tqdm import tqdm


from model import Dinov2ForSkySegmentation
from dataset import SkyDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")


    config = Dinov2Config.from_pretrained(args.model_name)
    model = Dinov2ForSkySegmentation.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)
    model.to(device)

    # Замораживаем DINOv2, обучаем только классификатор
    for name, param in model.named_parameters():
        if name.startswith("dinov2"):
            param.requires_grad = False

    # 2. Подготовка данных
    train_dataset = SkyDataset(split='train', image_size=args.image_size, cache_dir=args.cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 3. Настройка оптимизатора
    optimizer = AdamW(model.classifier.parameters(), lr=args.learning_rate)

    # 4. Цикл обучения
    model.train()
    for epoch in range(args.epochs):
        print(f"--- Эпоха {epoch + 1}/{args.epochs} ---")
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Обучение эпохи {epoch + 1}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Средняя ошибка за эпоху {epoch + 1}: {avg_loss:.4f}")

        # Сохранение чекпоинта
        checkpoint_path = os.path.join(args.output_dir, f"sky_detector_head_epoch_{epoch + 1}.pth")
        torch.save(model.classifier.state_dict(), checkpoint_path)
        print(f"Веса классификатора сохранены в {checkpoint_path}")

    print("Обучение завершено.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели для поиска неба")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base", help="Имя модели DINOv2")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох обучения")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Скорость обучения")
    parser.add_argument("--image_size", type=int, default=500, help="Размер изображения для обучения")
    parser.add_argument("--output_dir", type=str, default="../checkpoints", help="Директория для сохранения весов")
    parser.add_argument("--cache_dir", type=str, default="../data", help="Директория для кэша датасета")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)