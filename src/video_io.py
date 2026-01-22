import cv2
import queue
import threading
import time
from typing import Tuple


class VideoStream:
    """Класс для асинхронного чтения видео."""

    def __init__(self, source: str, queue_size: int = 128):
        """
        Args:
            source (str): Путь к видео.
            queue_size (int): Размер буфера кадров.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        """Запускает поток чтения."""
        self.thread.start()
        return self

    def _update(self):
        """Обновление буфера."""
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    return
                self.q.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        """Возвращает следующий кадр из буфера."""
        return self.q.get() if not self.q.empty() else None

    def more(self) -> bool:
        """Проверяет, есть ли еще кадры."""
        return not (self.stopped and self.q.empty())

    def stop(self):
        """Останавливает поток и освобождает ресурсы."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

    def get_info(self) -> Tuple[int, int, float, int]:
        """Возвращает метаданные видео: width, height, fps, total_frames."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return w, h, fps, total
