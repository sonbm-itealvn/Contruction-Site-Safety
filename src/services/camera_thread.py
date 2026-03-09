"""
Luồng camera hoặc video file: chạy detection mỗi N frame, giữa các lần chỉ vẽ lại box cũ (không tracker → nhẹ, không lag).
"""
import threading
import time
from typing import Callable, List, Optional, Union

import cv2
import numpy as np

from src.config.settings import CAMERA_INDEX, DETECT_EVERY_N_FRAMES
from src.models import DetectionResult, Violation
from src.services.detector import PPEDetector, draw_tracked_boxes


class CameraThread:
    """
    Chạy camera hoặc video file trong thread riêng, gọi callback khi có frame mới.
    - source: int = camera index (0 = webcam), str = đường dẫn file video (.mp4, .avi, ...).
    - on_video_end: gọi khi đọc hết file video (không dùng khi là camera).
    """

    def __init__(
        self,
        on_frame: Callable[[DetectionResult], None],
        source: Union[int, str] = CAMERA_INDEX,
        detector: Optional[PPEDetector] = None,
        on_video_end: Optional[Callable[[], None]] = None,
    ):
        self.on_frame = on_frame
        self.source = source  # int = camera index, str = video path
        self.detector = detector or PPEDetector()
        self.on_video_end = on_video_end
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._last_tracked_boxes: List[dict] = []
        self._last_violations: List[Violation] = []

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if isinstance(self.source, str):
            self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(int(self.source))
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
        is_video_file = isinstance(self.source, str)
        video_fps = self._cap.get(cv2.CAP_PROP_FPS) if is_video_file else 30
        frame_delay = 1.0 / video_fps if video_fps > 0 else 0.033
        detect_every = max(1, int(DETECT_EVERY_N_FRAMES))
        self._frame_index = 0
        self._last_tracked_boxes = []
        self._last_violations = []

        while not self._stop.is_set() and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                if is_video_file and callable(self.on_video_end):
                    try:
                        self.on_video_end()
                    except Exception:
                        pass
                break
            fps_frames += 1
            if fps_frames >= 10:
                fps_value = fps_frames / (time.perf_counter() - fps_start)
                fps_frames = 0
                fps_start = time.perf_counter()

            do_detect = self._frame_index % detect_every == 0 or not self._last_tracked_boxes
            if do_detect:
                frame_with_boxes, violations, tracked_boxes = self.detector.detect_and_draw_all(frame)
                self._last_violations = violations
                self._last_tracked_boxes = tracked_boxes
            else:
                # Không gọi YOLO, chỉ vẽ lại box cũ lên frame mới (rất nhẹ, không dùng tracker)
                if self._last_tracked_boxes:
                    frame_with_boxes = draw_tracked_boxes(frame, self._last_tracked_boxes, self._last_violations)
                else:
                    frame_with_boxes = frame
                    self._last_violations = []

            self._frame_index += 1
            result = DetectionResult(
                frame=frame_with_boxes,
                violations=self._last_violations,
                fps=fps_value,
            )
            try:
                self.on_frame(result)
            except Exception:
                pass
            time.sleep(frame_delay if is_video_file else 0.03)
        if self._cap:
            self._cap.release()
            self._cap = None
