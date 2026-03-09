"""
Cấu hình mặc định.
"""
import os

# Đường dẫn gốc dự án (thư mục chứa src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Assets
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
ICONS_DIR = os.path.join(ASSETS_DIR, "icons")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")

# Camera & detection
CAMERA_INDEX = 0  # 0 = webcam mặc định
CAMERA_AREA_NAME = "Khu vực A"
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng độ tin cậy (0-1)
# Model: None = dùng YOLO26 mặc định (yolo26n.pt), hoặc đường dẫn file .pt (model PPE tự train)
MODEL_PATH = None
# Tên model YOLO26 khi MODEL_PATH là None: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt
YOLO26_MODEL = "yolo26n.pt"

# Ghi hình
RECORDING_ENABLED = False
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "data", "recordings")
