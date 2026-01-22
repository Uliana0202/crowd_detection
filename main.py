import argparse
import cv2
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from src.video_io import VideoStream
from src.detector import CrowdDetector
from src.tracker import CrowdTracker
from src.visualizer import Visualizer


def set_seed(seed=42):
    """Для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Точка входа в программу."""
    set_seed()

    parser = argparse.ArgumentParser(description="Crowd Detection System")
    parser.add_argument("--source", type=str, default="media/crowd.mp4", help="Входное видео")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Веса модели")
    parser.add_argument("--output", type=str, default="output.mp4", help="Выходной файл")
    parser.add_argument("--sahi", action="store_true", help="Включить режим SAHI")

    args = parser.parse_args()

    stream = None
    out = None
    pbar = None

    try:
        detector = CrowdDetector(model_path=args.weights, use_sahi=args.sahi)
        visualizer = Visualizer()

        stream = VideoStream(args.source)
        width, height, fps, total_frames = stream.get_info()
        print(f"Видео: {width}x{height}, FPS: {fps:.2f}, Кадров: {total_frames}")

        tracker = CrowdTracker(fps=int(fps))

        # Настройка записи видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        stream.start()

        pbar = tqdm(total=total_frames, desc="Обработка", unit="frame")

        while stream.more():
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Получаем координаты людей
            detections = detector.detect(frame)

            # Связываем детекции между кадрами
            detections = tracker.update(detections)

            # Рисуем боксы и подписи
            class_names = detector.model.names
            annotated_frame = visualizer.draw(frame, detections, class_names)

            out.write(annotated_frame)
            pbar.update(1)

    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
    finally:
        if stream is not None:
            stream.stop()
        if out is not None:
            out.release()
        if pbar is not None:
            pbar.close()

        if Path(args.output).exists():
            print(f"\nРезультат сохранен в '{args.output}'")
        else:
            print("\nРезультат не сохранен!")


if __name__ == "__main__":
    main()
