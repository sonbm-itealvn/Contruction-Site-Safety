"""
Nhận diện vi phạm trang bị bảo hộ (mũ, áo) trên ảnh từ camera.
Sử dụng YOLOv26 (Ultralytics); hỗ trợ model PPE tùy chỉnh hoặc pretrained.
"""
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config.settings import CONFIDENCE_THRESHOLD, MODEL_PATH, YOLO26_MODEL
from src.models import Violation, ViolationType


class PPEDetector:
    """
    Phát hiện vi phạm PPE (thiếu mũ/áo bảo hộ) bằng YOLOv26.
    - Nếu MODEL_PATH được đặt: load file .pt (model PPE tự train).
    - Nếu MODEL_PATH = None: load YOLO26 pretrained (yolo26n.pt). Model COCO không có class PPE,
      nên không báo vi phạm; để nhận diện PPE thật hãy train model PPE và đặt MODEL_PATH.
    """

    # Nhãn class trong model PPE (sửa cho khớp với model của bạn)
    CLASS_NAMES = ["with_helmet", "without_helmet", "with_vest", "without_vest"]
    CLASS_TO_VIOLATION = {
        1: ViolationType.MISSING_HELMET,
        3: ViolationType.MISSING_VEST,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        yolo26_model: Optional[str] = None,
    ):
        self.model_path = model_path if model_path is not None else MODEL_PATH
        self.yolo26_model = yolo26_model or YOLO26_MODEL
        self.conf_threshold = confidence_threshold
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if self.model_path:
                self._model = YOLO(self.model_path)
                print(f"[PPEDetector] Đã load model PPE: {self.model_path}")
            else:
                self._model = YOLO(self.yolo26_model)
                print(f"[PPEDetector] Đang dùng YOLOv26: {self.yolo26_model}. Để nhận vi phạm PPE, đặt MODEL_PATH tới file model PPE.")
        except Exception as e:
            print(f"[PPEDetector] Lỗi load model: {e}. Chạy chế độ mẫu.")
            self._model = None

    def detect(self, frame: np.ndarray) -> List[Violation]:
        """
        Nhận diện vi phạm trên frame BGR.
        Trả về danh sách Violation (có bbox, type, confidence).
        """
        if self._model is not None:
            return self._detect_yolo(frame)
        return self._detect_demo(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Violation]:
        results = self._model.predict(frame, conf=self.conf_threshold, verbose=False, stream=False)
        violations = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.CLASS_TO_VIOLATION:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                violations.append(
                    Violation(
                        id=str(uuid.uuid4()),
                        violation_type=self.CLASS_TO_VIOLATION[cls_id],
                        confidence=conf,
                        timestamp=datetime.now(),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return violations

    def _detect_demo(self, frame: np.ndarray) -> List[Violation]:
        """
        Chế độ mẫu: tạo vài vi phạm giả định để test giao diện.
        Có thể thay bằng video/file ảnh thật khi đã có model.
        """
        h, w = frame.shape[:2]
        violations = []
        # Tạo 1–2 box mẫu ngẫu nhiên (vị trí cố định nhỏ để không lỗi)
        np.random.seed(hash(frame.tobytes()) % (2**32))
        for _ in range(np.random.randint(0, 3)):
            x1 = int(w * 0.2 + np.random.rand() * w * 0.3)
            y1 = int(h * 0.2 + np.random.rand() * h * 0.3)
            x2 = min(x1 + 80, w - 1)
            y2 = min(y1 + 120, h - 1)
            vt = ViolationType.MISSING_HELMET if np.random.rand() > 0.5 else ViolationType.MISSING_VEST
            violations.append(
                Violation(
                    id=str(uuid.uuid4()),
                    violation_type=vt,
                    confidence=0.85 + np.random.rand() * 0.1,
                    timestamp=datetime.now(),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return violations


def draw_violations_on_frame(
    frame: np.ndarray,
    violations: List[Violation],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ bounding box và nhãn vi phạm lên frame (BGR)."""
    out = frame.copy()
    for v in violations:
        label = v.label_vi.upper().replace(" ", " ")
        if v.bbox is None:
            continue
        x1, y1, x2, y2 = v.bbox
        color = (0, 0, 255) if v.violation_type == ViolationType.MISSING_HELMET else (0, 255, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {v.confidence_pct}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )
    return out
