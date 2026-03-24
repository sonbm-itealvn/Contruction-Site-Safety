"""
Cấu hình mặc định.
Có thể ghi đè bằng file data/settings.json (tạo qua màn hình Cài đặt).
"""
import json
import os

# Đường dẫn gốc dự án (thư mục chứa src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Assets
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
ICONS_DIR = os.path.join(ASSETS_DIR, "icons")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")

# File cài đặt do người dùng lưu (nếu có)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
USER_SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
CAMERAS_JSON_PATH = os.path.join(DATA_DIR, "cameras.json")

# Camera & detection (giá trị mặc định)
CAMERA_INDEX = 0
CAMERA_AREA_NAME = "Khu vực A"
CONFIDENCE_THRESHOLD = 0.25
VIOLATION_THROTTLE_SECONDS = 0.8
MODEL_PATH = os.path.join(PROJECT_ROOT, "helmet_best.pt")
YOLO26_MODEL = "yolo26n.pt"
# Chạy detection+tracking mỗi N frame; giữa các lần vẽ lại box cuối cùng.
DETECT_EVERY_N_FRAMES = 2
# Kích thước ảnh đưa vào YOLO (640 cho chất lượng tốt, 416 nếu máy yếu).
DETECT_IMGSZ = 640
# Sau bao nhiêu frame không thấy person thì xoá khỏi bộ nhớ tracking (cho phép thông báo lại nếu quay lại).
PERSON_STALE_FRAMES = 90
# Bật half precision (FP16) nếu có GPU CUDA → nhanh hơn ~2x.
USE_HALF_PRECISION = True

# Ghi hình
RECORDING_ENABLED = False
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "data", "recordings")


def load_user_settings():
    """Đọc data/settings.json (nếu có) và ghi đè lên biến cấu hình."""
    global CAMERA_INDEX, CAMERA_AREA_NAME, CONFIDENCE_THRESHOLD
    global VIOLATION_THROTTLE_SECONDS, MODEL_PATH, DETECT_IMGSZ, DETECT_EVERY_N_FRAMES
    if not os.path.isfile(USER_SETTINGS_PATH):
        return
    try:
        with open(USER_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "camera_index" in data:
            CAMERA_INDEX = int(data["camera_index"])
        if "camera_area_name" in data:
            CAMERA_AREA_NAME = str(data["camera_area_name"]).strip()
        if "confidence_threshold" in data:
            CONFIDENCE_THRESHOLD = float(data["confidence_threshold"])
        if "violation_throttle_seconds" in data:
            VIOLATION_THROTTLE_SECONDS = float(data["violation_throttle_seconds"])
        if "model_path" in data and data["model_path"]:
            MODEL_PATH = str(data["model_path"]).strip()
        if "detect_imgsz" in data:
            DETECT_IMGSZ = int(data["detect_imgsz"])
        if "detect_every_n_frames" in data:
            DETECT_EVERY_N_FRAMES = max(1, int(data["detect_every_n_frames"]))
    except Exception:
        pass


def load_cameras():
    """Danh sách camera: [{"name": str, "id": str, "source": int|str}, ...]. source = index máy hoặc URL RTSP."""
    if not os.path.isfile(CAMERAS_JSON_PATH):
        return [
            {"name": CAMERA_AREA_NAME, "id": "CAM-1", "source": CAMERA_INDEX},
            {"name": "Cổng chính", "id": "CAM-2", "source": 1},
            {"name": "Kho vật tư", "id": "CAM-3", "source": 2},
            {"name": "Khu vực B", "id": "CAM-4", "source": 3},
        ]
    try:
        with open(CAMERAS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return [{"name": CAMERA_AREA_NAME, "id": "CAM-1", "source": CAMERA_INDEX}]


def save_cameras(cameras: list):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CAMERAS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cameras, f, ensure_ascii=False, indent=2)


# Áp dụng cài đặt người dùng khi import
load_user_settings()
