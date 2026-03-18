"""
Hộp thoại Cài đặt — thiết kế chuẩn, đồng bộ với dashboard (pastel, card, bố cục rõ ràng).
"""
import json
import os
import sys

import customtkinter as ctk
from tkinter import filedialog

from src.config import settings

_FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"

_COLORS = {
    "bg": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "card_border": "#E2E8F0",
    "text_primary": "#1E293B",
    "text_secondary": "#64748B",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "btn_secondary": "#64748B",
    "input_border": "#CBD5E1",
}


class SettingsDialog(ctk.CTkToplevel):
    """
    Cài đặt: Camera, khu vực, ngưỡng nhận diện, throttle, đường dẫn model.
    """

    def __init__(self, parent, on_saved=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("Cài đặt hệ thống")
        self.geometry("520x620")
        self.resizable(True, True)
        self.configure(fg_color=_COLORS["bg"])
        self.on_saved = on_saved

        self._build_ui()
        self._load_current()
        self.transient(parent)
        self.grab_set()

    def _section(self, parent, title: str):
        """Tạo khung section: tiêu đề + frame nội dung."""
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(
            wrap,
            text=title,
            font=ctk.CTkFont(_FONT_FAMILY, size=13, weight="bold"),
            text_color=_COLORS["text_primary"],
        ).pack(anchor="w", pady=(0, 8))
        card = ctk.CTkFrame(
            wrap,
            fg_color=_COLORS["card_bg"],
            corner_radius=10,
            border_width=1,
            border_color=_COLORS["card_border"],
        )
        card.pack(fill="x")
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=12)
        return inner

    def _row(self, parent, label_text: str, hint: str = "", placeholder: str = ""):
        """Tạo một dòng: nhãn + ô nhập; trả về entry."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(
            row,
            text=label_text,
            font=ctk.CTkFont(_FONT_FAMILY, size=12),
            text_color=_COLORS["text_primary"],
        ).pack(anchor="w", pady=(0, 4))
        entry = ctk.CTkEntry(
            row,
            placeholder_text=placeholder or None,
            height=36,
            font=ctk.CTkFont(_FONT_FAMILY, size=12),
            border_color=_COLORS["input_border"],
            fg_color=_COLORS["card_bg"],
            corner_radius=8,
        )
        entry.pack(fill="x")
        if hint:
            ctk.CTkLabel(
                row,
                text=hint,
                font=ctk.CTkFont(_FONT_FAMILY, size=11),
                text_color=_COLORS["text_secondary"],
            ).pack(anchor="w", pady=(2, 0))
        return entry

    def _build_ui(self):
        # Nội dung cuộn được, nút cố định ở dưới
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=24, pady=20)

        # ---- Khung cuộn cho các section ----
        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ---- Header ----
        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(
            header,
            text="Cài đặt hệ thống",
            font=ctk.CTkFont(_FONT_FAMILY, size=20, weight="bold"),
            text_color=_COLORS["text_primary"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Camera, ngưỡng nhận diện và đường dẫn model YOLO",
            font=ctk.CTkFont(_FONT_FAMILY, size=12),
            text_color=_COLORS["text_secondary"],
        ).pack(anchor="w")

        # ---- Section 1: Camera & hiển thị ----
        s1 = self._section(scroll, "Camera & hiển thị")
        self._camera_index = self._row(s1, "Index camera", "0 = webcam mặc định, 1 = camera thứ 2...", "0")
        self._area_name = self._row(s1, "Tên khu vực", "Hiển thị trên giao diện chính", "Ví dụ: Khu vực A")

        # ---- Section 2: Ngưỡng nhận diện ----
        s2 = self._section(scroll, "Ngưỡng nhận diện")
        self._confidence = self._row(s2, "Confidence (0.1 – 0.9)", "Càng thấp càng nhạy", "0.25")
        self._throttle = self._row(s2, "Throttle (giây)", "Thời gian tối thiểu giữa hai lần ghi cùng loại vi phạm", "0.8")

        # ---- Section 3: Model ----
        s3 = self._section(scroll, "Model YOLO")
        self._model_path = self._row(s3, "Đường dẫn file .pt", "File model train 7 class PPE", "helmet_best.pt")
        ctk.CTkButton(
            s3,
            text="Chọn file...",
            width=110,
            height=34,
            fg_color=_COLORS["btn_secondary"],
            hover_color="#475569",
            font=ctk.CTkFont(_FONT_FAMILY, size=11, weight="bold"),
            command=self._pick_model_file,
        ).pack(anchor="w", pady=(6, 0))

        # ---- Nút cố định dưới cùng (ngoài scroll) ----
        sep = ctk.CTkFrame(main, fg_color=_COLORS["card_border"], height=1)
        sep.pack(fill="x", pady=(16, 12))
        btn_frame = ctk.CTkFrame(main, fg_color="transparent")
        btn_frame.pack(fill="x")
        ctk.CTkButton(
            btn_frame,
            text="Hủy",
            width=100,
            height=36,
            fg_color=_COLORS["btn_secondary"],
            hover_color="#475569",
            font=ctk.CTkFont(_FONT_FAMILY, size=12),
            command=self.destroy,
        ).pack(side="right", padx=(10, 0))
        ctk.CTkButton(
            btn_frame,
            text="Lưu cài đặt",
            width=120,
            height=36,
            fg_color=_COLORS["accent"],
            hover_color=_COLORS["accent_hover"],
            font=ctk.CTkFont(_FONT_FAMILY, size=12, weight="bold"),
            command=self._save,
        ).pack(side="right")

    def _load_current(self):
        self._camera_index.insert(0, str(settings.CAMERA_INDEX))
        self._area_name.insert(0, settings.CAMERA_AREA_NAME)
        self._confidence.insert(0, str(settings.CONFIDENCE_THRESHOLD))
        self._throttle.insert(0, str(settings.VIOLATION_THROTTLE_SECONDS))
        mp = settings.MODEL_PATH or ""
        if mp and os.path.isabs(mp):
            self._model_path.insert(0, mp)
        else:
            self._model_path.insert(0, os.path.basename(mp) if mp else "")

    def _pick_model_file(self):
        selected_path = filedialog.askopenfilename(
            title="Chọn file model (.pt)",
            filetypes=[
                ("PyTorch model", "*.pt"),
                ("Tất cả", "*.*"),
            ],
        )
        if not selected_path:
            return
        self._model_path.delete(0, "end")
        self._model_path.insert(0, selected_path)

    def _save(self):
        try:
            camera_index = int(self._camera_index.get().strip() or "0")
        except ValueError:
            camera_index = 0
        area_name = (self._area_name.get().strip() or settings.CAMERA_AREA_NAME)[:80]
        try:
            confidence = float(self._confidence.get().strip() or "0.25")
            confidence = max(0.1, min(0.95, confidence))
        except ValueError:
            confidence = settings.CONFIDENCE_THRESHOLD
        try:
            throttle = float(self._throttle.get().strip() or "0.8")
            throttle = max(0.2, min(10.0, throttle))
        except ValueError:
            throttle = settings.VIOLATION_THROTTLE_SECONDS
        model_path = (self._model_path.get().strip() or "").strip()
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(settings.PROJECT_ROOT, model_path)

        data = {
            "camera_index": camera_index,
            "camera_area_name": area_name,
            "confidence_threshold": confidence,
            "violation_throttle_seconds": throttle,
            "model_path": model_path or None,
            "track_person_only": getattr(settings, "TRACK_PERSON_ONLY", False),
        }
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        with open(settings.USER_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        settings.load_user_settings()
        if callable(self.on_saved):
            self.on_saved()
        self.destroy()
