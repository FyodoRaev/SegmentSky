import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt  # ### НОВОЕ ###


# --- Вспомогательные функции и классы ---

def calculate_iou(gt_mask, pred_mask):
    """Рассчитывает Intersection over Union (IoU) для бинарных масок."""
    gt_mask_bool = gt_mask > 0
    pred_mask_bool = pred_mask > 0
    intersection = np.logical_and(gt_mask_bool, pred_mask_bool).sum()
    union = np.logical_or(gt_mask_bool, pred_mask_bool).sum()
    if union == 0: return 1.0
    return intersection / union


# ### НОВАЯ ФУНКЦИЯ для метрик Precision и Recall ###
def calculate_precision_recall(gt_mask, pred_mask):
    """Рассчитывает Precision и Recall для класса 'небо'."""
    gt_mask_bool = gt_mask > 0
    pred_mask_bool = pred_mask > 0

    tp = np.logical_and(gt_mask_bool, pred_mask_bool).sum()  # True Positives
    fp = np.sum(np.logical_and(pred_mask_bool, ~gt_mask_bool))  # False Positives
    fn = np.sum(np.logical_and(~pred_mask_bool, gt_mask_bool))  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


# ### НОВАЯ ГЛАВНАЯ ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ ###
def create_comparison_visualization(original_image_path, pred_mask, gt_mask, metrics, output_path):
    """
    Создает и сохраняет комплексное изображение-отчет 2x2:
    - Предсказанная маска на оригинале
    - Эталонная маска на оригинале
    - График с метриками (IoU, Precision, Recall)
    - Текстовая сводка
    """
    iou, precision, recall = metrics['iou'], metrics['precision'], metrics['recall']

    # Загружаем оригинал
    original_img = cv2.imread(original_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Создаем визуализации (оригинал + маска)
    # Наложение для предсказания (красный цвет)
    pred_overlay = original_img.copy()
    pred_overlay[pred_mask > 0] = [255, 0, 0]
    pred_viz = cv2.addWeighted(original_img, 0.6, pred_overlay, 0.4, 0)

    # Наложение для эталона (зеленый цвет)
    gt_overlay = original_img.copy()
    gt_overlay[gt_mask > 0] = [0, 255, 0]
    gt_viz = cv2.addWeighted(original_img, 0.6, gt_overlay, 0.4, 0)

    # Создаем фигуру 2x2 для отчета
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle(f"Анализ сегментации для файла: {os.path.basename(original_image_path)}", fontsize=16)

    # 1. Левый верхний плот: Предсказание модели
    axs[0, 0].imshow(pred_viz)
    axs[0, 0].set_title('Предсказание модели (наложение красным)')
    axs[0, 0].axis('off')

    # 2. Правый верхний плот: Эталонная разметка
    axs[0, 1].imshow(gt_viz)
    axs[0, 1].set_title('Эталонная разметка (наложение зеленым)')
    axs[0, 1].axis('off')

    # 3. Левый нижний плот: Диаграмма метрик
    metric_names = ['IoU', 'Precision', 'Recall']
    metric_values = [iou, precision, recall]
    bars = axs[1, 0].barh(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axs[1, 0].set_title('Метрики качества')
    axs[1, 0].set_xlim(0, 1.05)
    axs[1, 0].invert_yaxis()  # IoU будет сверху
    # Добавляем значения на сами бары
    for bar in bars:
        width = bar.get_width()
        axs[1, 0].text(width + 0.01, bar.get_y() + bar.get_height() / 2., f'{width:.3f}', va='center')

    # 4. Правый нижний плот: Текстовая сводка
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Сводка')
    text_report = (
        f"• IoU: {iou:.4f}\n\n"
        f"• Precision: {precision:.4f}\n"
        f"  (Как хорошо модель избегает\n"
        f"  ложных срабатываний)\n\n"
        f"• Recall: {recall:.4f}\n"
        f"  (Как хорошо модель находит\n"
        f"  всю область неба)"
    )
    axs[1, 1].text(0.05, 0.5, text_report, ha='left', va='center', fontsize=12, wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Оставляем место для suptitle
    plt.savefig(output_path, dpi=150)
    plt.close(fig)  # Важно закрывать фигуру в цикле, чтобы не потреблять память


class SkySegmenter:
    # ... (Класс остается без изменений) ...
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-ade-640-640"):
        print("Инициализация модели... Это может занять некоторое время.")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Модель загружена и работает на устройстве: {self.device}")
        self.sky_class_id = 2

    def segment_sky(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1).cpu().numpy()[0]
        sky_mask = (pred_seg == self.sky_class_id).astype(np.uint8) * 255
        return sky_mask


# --- Логика выбора режима и обработки ---

def get_user_choice():
    # ... (Без изменений) ...
    while True:
        print("\nВыберите режим работы:")
        print("1: Простая обработка изображений (из папки 'fold_for_test_cases')")
        print("2: Тестирование на датасете с расчетом метрик (из 'fold_for_test_cases/images')")
        choice = input("Введите номер режима (1 или 2): ").strip()
        if choice in ['1', '2']:
            return choice
        else:
            print("! Неверный ввод. Пожалуйста, введите 1 или 2.")


def find_images_in_folder(folder_path):
    # ... (Без изменений) ...
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for extension in image_extensions: image_paths.extend(glob.glob(os.path.join(folder_path, extension)))
    return image_paths


def process_simple_images(segmenter):
    # ... (Без изменений) ...
    # Эта функция для режима 1 остается прежней
    source_folder, results_folder = "fold_for_test_cases", "results_simple"
    print(f"\n--- Режим 1: Простая обработка из '{source_folder}' ---")
    image_paths = find_images_in_folder(source_folder)
    if not image_paths:
        print(f"Ошибка: В папке '{source_folder}' не найдено изображений.")
        return
    masks_dest_folder = os.path.join(results_folder, "masks")
    vis_dest_folder = os.path.join(results_folder, "visualizations")
    os.makedirs(masks_dest_folder, exist_ok=True)
    os.makedirs(vis_dest_folder, exist_ok=True)
    print(f"Найдено {len(image_paths)} изображений. Результаты будут в '{results_folder}'.")
    for i, image_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nОбработка {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
        try:
            predicted_mask = segmenter.segment_sky(image_path)
            # Для простой визуализации используем метод из класса
            original_viz = cv2.imread(image_path)
            overlay = original_viz.copy()
            overlay[predicted_mask > 0] = [0, 0, 255]  # Синее наложение
            visualization = cv2.addWeighted(original_viz, 0.7, overlay, 0.3, 0)

            mask_path = os.path.join(masks_dest_folder, f"{base_name}_mask.png")
            vis_path = os.path.join(vis_dest_folder, f"{base_name}_visualization.png")
            cv2.imwrite(mask_path, predicted_mask)
            cv2.imwrite(vis_path, visualization)
            print(f"  ✓ Результаты сохранены.")
        except Exception as e:
            print(f"  ✗ Ошибка при обработке: {e}")
    print("\nОбработка завершена.")


# ### ОБНОВЛЕННАЯ ФУНКЦИЯ ДЛЯ РЕЖИМА 2 ###
def process_dataset_with_metrics(segmenter):
    """Режим 2: Тестирование на датасете с созданием информативных отчетов."""
    images_folder = os.path.join("fold_for_test_cases", "images")
    gt_masks_folder = os.path.join("fold_for_test_cases", "masks_ground_truth")
    results_folder = "results_with_metrics"  # Папка для комплексных отчетов

    print(f"\n--- Режим 2: Тестирование с созданием отчетов из '{images_folder}' ---")

    image_paths = find_images_in_folder(images_folder)
    if not image_paths:
        print(f"Ошибка: В папке '{images_folder}' не найдено изображений.")
        return

    os.makedirs(results_folder, exist_ok=True)
    all_metrics = []

    print(f"Найдено {len(image_paths)} изображений. Отчеты будут сохранены в '{results_folder}'.")

    for i, image_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nТестирование {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
        try:
            # Предсказание
            predicted_mask = segmenter.segment_sky(image_path)

            # Загрузка эталонной маски
            gt_mask_path = os.path.join(gt_masks_folder, f"{base_name}_mask.png")
            if not os.path.exists(gt_mask_path):
                print(f"  ! Предупреждение: Эталонная маска не найдена, пропуск.")
                continue
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

            # Расчет всех метрик
            iou = calculate_iou(gt_mask, predicted_mask)
            precision, recall = calculate_precision_recall(gt_mask, predicted_mask)

            current_metrics = {'iou': iou, 'precision': precision, 'recall': recall}
            all_metrics.append(current_metrics)
            print(f"  ✓ Метрики: IoU={iou:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

            # Создание и сохранение комплексного отчета
            report_path = os.path.join(results_folder, f"{base_name}_report.png")
            create_comparison_visualization(image_path, predicted_mask, gt_mask, current_metrics, report_path)
            print(f"  ✓ Информативный отчет сохранен в '{results_folder}'")

        except Exception as e:
            print(f"  ✗ Ошибка при обработке: {e}")

    if all_metrics:
        # Считаем средние значения по всем метрикам
        avg_iou = np.mean([m['iou'] for m in all_metrics])
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])

        print("\n" + "=" * 60)
        print("Тестирование завершено!")
        print(f"Средние метрики по {len(all_metrics)} изображениям:")
        print(f"  - Средний IoU:       {avg_iou:.4f}")
        print(f"  - Средняя Precision: {avg_precision:.4f}")
        print(f"  - Средняя Recall:    {avg_recall:.4f}")
        print("=" * 60)


def main():
    choice = get_user_choice()
    segmenter = SkySegmenter()
    if choice == '1':
        process_simple_images(segmenter)
    elif choice == '2':
        process_dataset_with_metrics(segmenter)


if __name__ == "__main__":
    main()