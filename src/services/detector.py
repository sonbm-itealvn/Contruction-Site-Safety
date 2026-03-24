"""
Nhận diện vi phạm PPE + tracking bằng ByteTrack (qua ultralytics model.track).
Dataset nhãn: "Gloves", "Helmet", "Non-Helmet", "Person", "Shoes", "Vest", "bare-arms".
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.config.settings import (
    CONFIDENCE_THRESHOLD,
    DETECT_IMGSZ,
    MODEL_PATH,
    USE_HALF_PRECISION,
    YOLO26_MODEL,
)
from src.models import Violation, ViolationType

DATASET_CLASS_NAMES = [
    "Gloves",      # 0
    "Helmet",      # 1
    "Non-Helmet",  # 2  → vi phạm
    "Person",      # 3
    "Shoes",       # 4
    "Vest",        # 5
    "bare-arms",   # 6  → vi phạm
]

PERSON_CLS_ID = 3

CLASS_INDEX_TO_VIOLATION = {
    2: ViolationType.MISSING_HELMET,   # Non-Helmet
    6: ViolationType.MISSING_VEST,     # bare-arms
}

BOX_COLORS = {
    0: (255, 200, 0),   # Gloves - cyan-ish
    1: (0, 200, 0),     # Helmet - green
    2: (0, 0, 255),     # Non-Helmet - red
    3: (255, 180, 0),   # Person - orange
    4: (200, 200, 0),   # Shoes - teal
    5: (0, 255, 200),   # Vest - green-cyan
    6: (0, 80, 255),    # bare-arms - dark red
}

VIOLATION_COLOR = (0, 0, 255)
PERSON_OK_COLOR = (0, 200, 0)
PERSON_VIOLATION_COLOR = (0, 0, 255)


@dataclass
class TrackingResult:
    """Kết quả tracking một frame."""
    persons: List[dict] = field(default_factory=list)
    all_boxes: List[dict] = field(default_factory=list)
    person_violations: Dict[int, Set[ViolationType]] = field(default_factory=dict)


class PPEDetector:
    """
    Phát hiện PPE + tracking ByteTrack.
    - track_frame(): chạy model.track(persist=True), trả TrackingResult với person IDs
    - draw_tracking_frame(): vẽ bounding box + person ID lên frame
    """

    CLASS_NAMES = DATASET_CLASS_NAMES
    CLASS_TO_VIOLATION = CLASS_INDEX_TO_VIOLATION

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
        self._names_list: List[str] = list(DATASET_CLASS_NAMES)
        self._use_half = False
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if self.model_path:
                self._model = YOLO(self.model_path)
                print(f"[PPEDetector] Đã load model PPE: {self.model_path}")
            else:
                self._model = YOLO(self.yolo26_model)
                print(f"[PPEDetector] YOLOv26: {self.yolo26_model}")

            class_names = getattr(self._model, "names", None) or DATASET_CLASS_NAMES
            if isinstance(class_names, dict):
                max_id = max(class_names.keys()) if class_names else 0
                self._names_list = [class_names.get(i, str(i)) for i in range(max_id + 1)]
            else:
                self._names_list = list(class_names) if class_names else []

            if USE_HALF_PRECISION:
                import torch
                if torch.cuda.is_available():
                    self._use_half = True
                    print("[PPEDetector] GPU CUDA detected → FP16 enabled")
                else:
                    print("[PPEDetector] No CUDA → FP16 disabled, using CPU")
        except ImportError:
            pass
        except Exception as e:
            print(f"[PPEDetector] Lỗi load model: {e}")
            self._model = None

    # ------------------------------------------------------------------
    # Tracking API (chính)
    # ------------------------------------------------------------------

    def track_frame(self, frame: np.ndarray) -> TrackingResult:
        """
        Chạy model.track(persist=True) → ByteTrack giữ ID liên tục.
        Trả về TrackingResult: persons (với track_id), all_boxes, person_violations.
        """
        if self._model is None:
            return TrackingResult()

        results = self._model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            imgsz=DETECT_IMGSZ,
            verbose=False,
            stream=False,
            half=self._use_half,
        )

        persons: List[dict] = []
        violation_detections: List[dict] = []
        all_boxes: List[dict] = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                name = self._names_list[cls_id] if cls_id < len(self._names_list) else str(cls_id)

                info = {
                    "cls_id": cls_id, "conf": conf, "track_id": track_id,
                    "bbox": (x1, y1, x2, y2), "name": name,
                }
                all_boxes.append(info)

                if cls_id == PERSON_CLS_ID:
                    persons.append(info)
                elif cls_id in CLASS_INDEX_TO_VIOLATION:
                    violation_detections.append(info)

        person_violations = self._associate_violations_to_persons(
            violation_detections, persons
        )

        return TrackingResult(
            persons=persons,
            all_boxes=all_boxes,
            person_violations=person_violations,
        )

    def _associate_violations_to_persons(
        self,
        violation_detections: List[dict],
        persons: List[dict],
    ) -> Dict[int, Set[ViolationType]]:
        """Gán mỗi violation box cho Person gần nhất (IoU / containment)."""
        result: Dict[int, Set[ViolationType]] = {}
        if not persons or not violation_detections:
            return result

        for vd in violation_detections:
            vx1, vy1, vx2, vy2 = vd["bbox"]
            vcx, vcy = (vx1 + vx2) // 2, (vy1 + vy2) // 2

            best_pid = None
            best_score = -1.0

            for p in persons:
                pid = p["track_id"]
                if pid < 0:
                    continue
                px1, py1, px2, py2 = p["bbox"]

                # Ưu tiên: tâm violation nằm trong person box
                inside = px1 <= vcx <= px2 and py1 <= vcy <= py2

                # Tính IoU
                ix1, iy1 = max(vx1, px1), max(vy1, py1)
                ix2, iy2 = min(vx2, px2), min(vy2, py2)
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                else:
                    inter = 0
                va = max(1, (vx2 - vx1) * (vy2 - vy1))
                pa = max(1, (px2 - px1) * (py2 - py1))
                iou = inter / (va + pa - inter) if (va + pa - inter) > 0 else 0

                score = iou + (1.0 if inside else 0.0)
                if score > best_score:
                    best_score = score
                    best_pid = pid

            if best_pid is not None and best_score > 0.05:
                vtype = CLASS_INDEX_TO_VIOLATION[vd["cls_id"]]
                result.setdefault(best_pid, set()).add(vtype)

        return result

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_tracking_frame(
        self,
        frame: np.ndarray,
        all_boxes: List[dict],
        person_violations: Dict[int, Set[ViolationType]],
    ) -> np.ndarray:
        """Vẽ tất cả box lên frame. Person box có ID và trạng thái vi phạm."""
        out = frame.copy()
        for d in all_boxes:
            x1, y1, x2, y2 = d["bbox"]
            cls_id = d["cls_id"]
            conf = d["conf"]
            track_id = d.get("track_id", -1)
            name = d.get("name", "")

            if cls_id == PERSON_CLS_ID:
                has_violation = track_id >= 0 and track_id in person_violations
                color = PERSON_VIOLATION_COLOR if has_violation else PERSON_OK_COLOR
                thickness = 3 if has_violation else 2

                cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

                if track_id >= 0:
                    id_label = f"ID-{track_id}"
                    if has_violation:
                        vtypes = person_violations[track_id]
                        vnames = []
                        if ViolationType.MISSING_HELMET in vtypes:
                            vnames.append("Thieu mu")
                        if ViolationType.MISSING_VEST in vtypes:
                            vnames.append("Thieu ao")
                        id_label += " " + "+".join(vnames)

                    (tw, th), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(out, id_label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif cls_id in CLASS_INDEX_TO_VIOLATION:
                cv2.rectangle(out, (x1, y1), (x2, y2), VIOLATION_COLOR, 2)
                label = f"{name} {int(conf*100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), VIOLATION_COLOR, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                color = BOX_COLORS.get(cls_id, (200, 200, 200))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
                label = f"{name} {int(conf*100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return out

    # ------------------------------------------------------------------
    # Legacy API (giữ để backward-compat nếu cần)
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Violation]:
        if self._model is not None:
            return self._detect_yolo(frame)
        return []

    def _detect_yolo(self, frame: np.ndarray) -> List[Violation]:
        results = self._model.predict(
            frame, conf=self.conf_threshold, verbose=False, stream=False, imgsz=DETECT_IMGSZ
        )
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
