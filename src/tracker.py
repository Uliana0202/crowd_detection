import supervision as sv

class CrowdTracker:
    """Обертка над алгоритмом ByteTrack для отслеживания объектов."""

    def __init__(self, fps: int = 30):
        """
        Args:
            fps (int): Кадровая частота видео.
        """
        # Настройки ByteTrack оптимизированы для толпы
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, # Порог подтверждения трека
            lost_track_buffer=30,            # Сколько кадров помнить потерянный объект
            minimum_matching_threshold=0.8,  # Порог совпадения (IoU)
            frame_rate=fps
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Обновляет треки на основе новых детекций."""
        return self.tracker.update_with_detections(detections)
