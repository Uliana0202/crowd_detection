import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO


class CrowdDetector:
    """
    Класс детекции, поддерживающий стандартный инференс и SAHI (Slicing).
    """

    def __init__(self, model_path: str, use_sahi: bool = False):
        """
        Args:
            model_path (str): Путь к .pt файлу (best.pt).
            use_sahi (bool): Включить ли нарезку кадров.
        """
        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = 'cpu'

        self.model = YOLO(model_path).to(self.device)
        self.use_sahi = use_sahi

        if self.use_sahi:
            self.slicer = sv.InferenceSlicer(
                callback=self._callback,
                slice_wh=(640, 640),
                iou_threshold=0.5
            )

    def _callback(self, image_slice: np.ndarray) -> sv.Detections:
        """Callback функция для SAHI."""
        result = self.model(image_slice, verbose=False, conf=0.3, device=self.device)[0]
        return sv.Detections.from_ultralytics(result)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Выполняет детекцию на кадре."""
        if self.use_sahi:
            return self.slicer(frame)
        else:
            result = self.model(frame, verbose=False, conf=0.3, iou=0.7, device=self.device)[0]
            return sv.Detections.from_ultralytics(result)
