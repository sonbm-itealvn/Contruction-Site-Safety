"""
Luồng camera: đọc frame từ camera, chạy detection, gửi kết quả lên UI.
"""
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from src.config.settings import CAMERA_INDEX
from src.models import DetectionResult, Violation
from src.services.detector import PPEDetector, draw_violations_on_frame


class CameraThread:
    """Chạy camera và detection trong thread riêng, gọi callback khi có frame mới."""

    def __init__(
        self,
        on_frame: Callable[[DetectionResult], None],
        camera_index: int = CAMERA_INDEX,
        detector: Optional[PPEDetector] = None,
    ):
        self.on_frame = on_frame
        self.camera_index = camera_index
        self.detector = detector or PPEDetector()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self.on_frame(DetectionResult(frame=np.zeros((480, 640, 3), dtype=np.uint8), violations=[], fps=0))
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def _run(self):
        fps_start = time.perf_counter()
        fps_frames = 0
        fps_value = 0.0
        while not self._stop.is_set() and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            fps_frames += 1
            if fps_frames >= 10:
                fps_value = fps_frames / (time.perf_counter() - fps_start)
                fps_frames = 0
                fps_start = time.perf_counter()

            violations = self.detector.detect(frame)
            frame_with_boxes = draw_violations_on_frame(frame, violations)
            result = DetectionResult(
                frame=frame_with_boxes,
                violations=violations,
                fps=fps_value,
            )
            try:
                self.on_frame(result)
            except Exception:
                pass
            time.sleep(0.03)
        if self._cap:
            self._cap.release()
            self._cap = None
