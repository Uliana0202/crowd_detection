import numpy as np
import supervision as sv


class Visualizer:
    """Класс для отрисовки результатов детекции."""

    def __init__(self):
        # Линии бокса
        self.box_annotator = sv.BoxAnnotator(
            thickness=2
        )
        # Подписи с фоном
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=5
        )

    def draw(self, frame: np.ndarray, detections: sv.Detections, class_names: dict) -> np.ndarray:
        """Рисует боксы и подписи."""
        annotated_frame = frame.copy()

        # Рисуем боксы
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        # Формируем подписи
        labels = []
        for class_id, confidence, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id):
            class_name = class_names.get(class_id, "Unknown")

            # Можно добавить tracker_id для распознанного человека
            # label = f"#{tracker_id} {class_name} {confidence:.2f}"

            label = f"{class_name} {confidence:.2f}"
            labels.append(label)

        # Рисуем подписи
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        return annotated_frame
