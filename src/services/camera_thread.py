"""
Luồng camera / video: detection + ByteTrack tracking.
- Mỗi người được gán ID duy nhất (ByteTrack qua model.track).
- Vi phạm chỉ thông báo 1 lần cho mỗi (person_id, violation_type).
- Khi người biến mất (stale) → xoá khỏi bộ nhớ; nếu quay lại sẽ thông báo lại.
"""
import threading
import time
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Union

import cv2
import numpy as np

from src.config.settings import (
    CAMERA_INDEX,
    DETECT_EVERY_N_FRAMES,
    PERSON_STALE_FRAMES,
)
from src.models import DetectionResult, Violation, ViolationType
from src.services.detector import PPEDetector, TrackingResult


class CameraThread:
    """
    Chạy camera / video trong thread riêng.
    - source: int (camera index) hoặc str (đường dẫn video file).
    - Tracking: model.track(persist=True) → ByteTrack giữ person ID liên tục.
    - Violation dedup: mỗi person_id chỉ thông báo 1 lần / loại vi phạm.
    """

    def __init__(
        self,
        on_frame: Callable[[DetectionResult], None],
        source: Union[int, str] = CAMERA_INDEX,
        detector: Optional[PPEDetector] = None,
        on_video_end: Optional[Callable[[], None]] = None,
    ):
        self.on_frame = on_frame
        self.source = source
        self.detector = detector or PPEDetector()
        self.on_video_end = on_video_end
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if isinstance(self.source, str):
            self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(int(self.source))
        if not self._cap.isOpened():
            self.on_frame(DetectionResult(
                frame=np.zeros((480, 640, 3), dtype=np.uint8), violations=[], fps=0,
            ))
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

        frame_index = 0
        last_tracking: Optional[TrackingResult] = None

        # Per-person violation state
        notified: Dict[int, Set[ViolationType]] = {}
        person_last_seen: Dict[int, int] = {}

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

            do_detect = (frame_index % detect_every == 0) or last_tracking is None
            new_violations: List[Violation] = []

            if do_detect:
                tracking = self.detector.track_frame(frame)
                last_tracking = tracking

                for p in tracking.persons:
                    pid = p["track_id"]
                    if pid < 0:
                        continue
                    person_last_seen[pid] = frame_index

                # Check violations — chỉ thông báo lần đầu cho mỗi (person_id, vtype)
                for pid, vtypes in tracking.person_violations.items():
                    if pid < 0:
                        continue
                    if pid not in notified:
                        notified[pid] = set()
                    for vtype in vtypes:
                        if vtype not in notified[pid]:
                            notified[pid].add(vtype)
                            best_conf = 0.6
                            best_bbox = None
                            for p in tracking.persons:
                                if p["track_id"] == pid:
                                    best_bbox = p["bbox"]
                                    best_conf = p["conf"]
                                    break
                            new_violations.append(Violation(
                                id=str(uuid.uuid4()),
                                violation_type=vtype,
                                confidence=best_conf,
                                timestamp=datetime.now(),
                                bbox=best_bbox,
                                person_id=pid,
                            ))

                # Cleanup stale persons
                stale_ids = [
                    pid for pid, last in person_last_seen.items()
                    if frame_index - last > PERSON_STALE_FRAMES
                ]
                for pid in stale_ids:
                    person_last_seen.pop(pid, None)
                    notified.pop(pid, None)

                drawn = self.detector.draw_tracking_frame(
                    frame, tracking.all_boxes, tracking.person_violations,
                )
            else:
                if last_tracking is not None and last_tracking.all_boxes:
                    drawn = self.detector.draw_tracking_frame(
                        frame, last_tracking.all_boxes, last_tracking.person_violations,
                    )
                else:
                    drawn = frame

            frame_index += 1
            result = DetectionResult(frame=drawn, violations=new_violations, fps=fps_value)
            try:
                self.on_frame(result)
            except Exception:
                pass
            time.sleep(frame_delay if is_video_file else 0.03)

        if self._cap:
            self._cap.release()
            self._cap = None
