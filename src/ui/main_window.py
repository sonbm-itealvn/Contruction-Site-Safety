"""
Cửa sổ chính - Dashboard giám sát an toàn lao động.
Giao diện: header, KPI cards, cảnh báo, camera trực tiếp, danh sách vi phạm.
"""
import json
import os
from collections import deque
from dataclasses import replace
from datetime import datetime
from typing import Deque, List

import cv2
import customtkinter as ctk
from PIL import Image
import numpy as np

from src.config.settings import (
    CAMERA_AREA_NAME,
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    DATA_DIR,
    MODEL_PATH,
    PROJECT_ROOT,
    USER_SETTINGS_PATH,
    VIOLATION_THROTTLE_SECONDS,
    load_cameras,
    load_user_settings,
    save_cameras,
)
from src.models import Violation, ViolationType, DetectionResult
from src.services.camera_thread import CameraThread
from src.ui.settings_dialog import SettingsDialog

# Giao diện sáng, tông nhẹ nhàng và hơi trong (pastel)
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
import sys
_FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"

_COLORS = {
    "bg_app": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "total_bg": "#FFEBEE",
    "total_accent": "#EB5757",
    "helmet_bg": "#FFF8F3",
    "helmet_accent": "#E67E22",
    "vest_bg": "#FFFBF3",
    "vest_accent": "#D4A017",
    "both_bg": "#FFF3F3",
    "both_accent": "#C0392B",
    "alert_bg": "#FEE8E8",
    "alert_border": "#EB5757",
    "text_primary": "#1E293B",
    "text_secondary": "#64748B",
}


class KPICard(ctk.CTkFrame):
    """Thẻ KPI: nền pastel nhẹ, icon + số màu accent."""

    def __init__(
        self,
        master,
        title: str,
        value: int = 0,
        bg_color: str = "#FFEBEE",
        accent_color: str = "#EB5757",
        icon_text: str = "⚠",
        **kwargs,
    ):
        super().__init__(master, fg_color=bg_color, corner_radius=12, **kwargs)
        self._value_var = ctk.StringVar(value=str(value))
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=12)
        ctk.CTkLabel(
            inner, text=icon_text, font=ctk.CTkFont(_FONT_FAMILY, size=26), text_color=accent_color,
        ).pack(anchor="w")
        ctk.CTkLabel(
            inner, textvariable=self._value_var, font=ctk.CTkFont(_FONT_FAMILY, size=26, weight="bold"), text_color=accent_color,
        ).pack(anchor="w")
        ctk.CTkLabel(
            inner, text=title, font=ctk.CTkFont(_FONT_FAMILY, size=13), text_color=_COLORS["text_secondary"],
        ).pack(anchor="w")

    def set_value(self, value: int):
        self._value_var.set(str(value))


class ViolationCard(ctk.CTkFrame):
    """Thẻ vi phạm: nền pastel nhẹ, viền mỏng."""

    def __init__(self, master, violation: Violation, **kwargs):
        if violation.violation_type == ViolationType.MISSING_BOTH:
            bg_color, accent = _COLORS["both_bg"], _COLORS["both_accent"]
        elif violation.violation_type == ViolationType.MISSING_HELMET:
            bg_color, accent = _COLORS["helmet_bg"], _COLORS["helmet_accent"]
        else:
            bg_color, accent = _COLORS["vest_bg"], _COLORS["vest_accent"]
        super().__init__(master, fg_color=bg_color, corner_radius=10, border_width=1, border_color="#E2E8F0", **kwargs)
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=12, pady=10)
        icon = "⛑🦺" if violation.violation_type == ViolationType.MISSING_BOTH else ("⛑" if violation.violation_type == ViolationType.MISSING_HELMET else "🦺")
        ctk.CTkLabel(inner, text=icon, font=ctk.CTkFont(_FONT_FAMILY, size=20), text_color=accent).pack(side="left", padx=(0, 10))
        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)
        person_tag = f" [{violation.person_label}]" if getattr(violation, "person_id", None) is not None else ""
        ctk.CTkLabel(
            right, text=f"{violation.label_vi}{person_tag}",
            font=ctk.CTkFont(_FONT_FAMILY, size=14, weight="bold"), text_color=_COLORS["text_primary"], anchor="w",
        ).pack(anchor="w")
        loc = f" - {violation.location}" if getattr(violation, "location", None) else ""
        ctk.CTkLabel(
            right, text=f"🕐 {violation.time_str}{loc} · {violation.confidence_pct}",
            font=ctk.CTkFont(_FONT_FAMILY, size=12), text_color=_COLORS["text_secondary"], anchor="w",
        ).pack(anchor="w")


class MainWindow:
    """Dashboard giám sát an toàn lao động - nhận diện real-time qua camera."""

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Hệ thống giám sát an toàn lao động")
        self.root.geometry("1280x780")
        self.root.minsize(1024, 600)
        self.root.configure(fg_color="#F8FAFC")

        # Lưu vi phạm để hiển thị (giới hạn 100)
        self._violations: Deque[Violation] = deque(maxlen=100)
        self._camera_thread: CameraThread | None = None
        self._video_image: ctk.CTkImage | None = None  # Giữ reference để cập nhật
        self._video_label: ctk.CTkLabel | None = None
        self._last_violation_time: dict = {}  # throttle: type -> last add time
        # Nguồn hiện tại: int = camera index, str = đường dẫn file video
        self._current_source: int | str = CAMERA_INDEX
        # Danh sách camera (từ data/cameras.json); index đang chọn trên dashboard
        self._cameras = load_cameras()
        self._current_camera_index = 0
        self._is_playing_video_file = False
        if self._cameras:
            self._current_source = self._cameras[0].get("source", CAMERA_INDEX)

        self._setup_ui()
        self._start_source()

    def _setup_ui(self):
        # ---- Sidebar: 3 nút rõ chữ (Trang chủ, Camera, Cài đặt) ----
        sidebar = ctk.CTkFrame(self.root, fg_color="#E5E7EB", width=100, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ctk.CTkLabel(sidebar, text="Menu", font=ctk.CTkFont(_FONT_FAMILY, size=11, weight="bold"), text_color=_COLORS["text_secondary"]).pack(pady=(16, 10))
        self._btn_dashboard = ctk.CTkButton(
            sidebar, text="Trang chủ", width=84, height=44, corner_radius=10,
            fg_color="#1E293B", hover_color="#374151", text_color="white",
            font=ctk.CTkFont(_FONT_FAMILY, size=12), command=lambda: self._show_page("dashboard"),
        )
        self._btn_dashboard.pack(pady=6)
        self._btn_cameras = ctk.CTkButton(
            sidebar, text="Camera", width=84, height=44, corner_radius=10,
            fg_color="transparent", hover_color="#D1D5DB", text_color=_COLORS["text_primary"],
            font=ctk.CTkFont(_FONT_FAMILY, size=12), command=lambda: self._show_page("cameras"),
        )
        self._btn_cameras.pack(pady=6)
        self._btn_settings = ctk.CTkButton(
            sidebar, text="Cài đặt", width=84, height=44, corner_radius=10,
            fg_color="transparent", hover_color="#D1D5DB", text_color=_COLORS["text_primary"],
            font=ctk.CTkFont(_FONT_FAMILY, size=12), command=lambda: self._show_page("settings"),
        )
        self._btn_settings.pack(pady=6)
        ctk.CTkLabel(sidebar, text="").pack(fill="y", expand=True)
        ctk.CTkButton(sidebar, text="Tài khoản", width=84, height=36, corner_radius=10, fg_color="transparent", hover_color="#D1D5DB", font=ctk.CTkFont(_FONT_FAMILY, size=11)).pack(pady=(0, 16))

        self._content = ctk.CTkFrame(self.root, fg_color="transparent")
        self._content.pack(side="left", fill="both", expand=True, padx=20, pady=16)
        self._dashboard_frame = ctk.CTkFrame(self._content, fg_color="transparent")
        self._cameras_frame = ctk.CTkFrame(self._content, fg_color="transparent")
        self._settings_frame = ctk.CTkFrame(self._content, fg_color="transparent")

        # ---- KPI + LIVE ----
        kpi_frame = ctk.CTkFrame(self._dashboard_frame, fg_color="transparent")
        kpi_frame.pack(fill="x", padx=24, pady=(0, 12))
        self._kpi_total = KPICard(kpi_frame, "Tổng vi phạm", 0, bg_color=_COLORS["total_bg"], accent_color=_COLORS["total_accent"], icon_text="⚠")
        self._kpi_total.pack(side="left", fill="x", expand=True, padx=4)
        self._kpi_helmet = KPICard(kpi_frame, "Thiếu mũ bảo hộ", 0, bg_color=_COLORS["helmet_bg"], accent_color=_COLORS["helmet_accent"], icon_text="⛑")
        self._kpi_helmet.pack(side="left", fill="x", expand=True, padx=4)
        self._kpi_vest = KPICard(kpi_frame, "Thiếu áo bảo hộ", 0, bg_color=_COLORS["vest_bg"], accent_color=_COLORS["vest_accent"], icon_text="🦺")
        self._kpi_vest.pack(side="left", fill="x", expand=True, padx=4)
        self._kpi_both = KPICard(kpi_frame, "Thiếu cả hai", 0, bg_color=_COLORS["both_bg"], accent_color=_COLORS["both_accent"], icon_text="⛑🦺")
        self._kpi_both.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkLabel(kpi_frame, text="● TRỰC TIẾP", font=ctk.CTkFont(_FONT_FAMILY, size=12, weight="bold"), text_color="#16A34A").pack(side="right", padx=12)

        # ---- Alert bar ----
        self._alert_bar = ctk.CTkFrame(self._dashboard_frame, fg_color=_COLORS["alert_bg"], corner_radius=8, height=48, border_width=0)
        self._alert_bar.pack(fill="x", pady=(0, 12))
        # Viền trái đỏ (frame mỏng)
        alert_left_border = ctk.CTkFrame(self._alert_bar, fg_color=_COLORS["alert_border"], width=4, corner_radius=2)
        alert_left_border.place(relx=0, rely=0, relheight=1, anchor="nw")
        inner_alert = ctk.CTkFrame(self._alert_bar, fg_color="transparent")
        inner_alert.place(relx=0.5, rely=0.5, anchor="center")
        self._alert_label = ctk.CTkLabel(
            inner_alert, text="Không có vi phạm.",
            font=ctk.CTkFont(_FONT_FAMILY, size=13), text_color=_COLORS["alert_border"],
        )
        self._alert_label.pack()
        self._alert_bar.pack_forget()

        self._alert_bar_placeholder = ctk.CTkFrame(self._dashboard_frame, fg_color="transparent", height=4)
        self._alert_bar_placeholder.pack(fill="x")

        # ---- Content: Video (có nút trái/phải) + List ----
        content = ctk.CTkFrame(self._dashboard_frame, fg_color="transparent")
        content.pack(fill="both", expand=True)

        left_pane = ctk.CTkFrame(content, fg_color="transparent")
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 12))
        video_row = ctk.CTkFrame(left_pane, fg_color="transparent")
        video_row.pack(fill="both", expand=True)
        self._btn_prev_cam = ctk.CTkButton(video_row, text="‹", width=40, height=40, corner_radius=20, fg_color="#E2E8F0", hover_color="#CBD5E1", font=ctk.CTkFont(_FONT_FAMILY, size=24), command=self._prev_camera)
        self._btn_prev_cam.pack(side="left", padx=(0, 8), pady=40)
        self._video_container = ctk.CTkFrame(video_row, fg_color="#1E293B", corner_radius=12)
        self._video_container.pack(side="left", fill="both", expand=True)
        self._video_label = ctk.CTkLabel(
            self._video_container, text="Đang kết nối camera...",
            font=ctk.CTkFont(_FONT_FAMILY, size=14), text_color="gray",
        )
        self._video_label.pack(fill="both", expand=True, padx=8, pady=8)
        self._live_badge = ctk.CTkLabel(
            self._video_container, text=" TRỰC TIẾP ", font=ctk.CTkFont(_FONT_FAMILY, size=11, weight="bold"),
            fg_color="black", text_color="red", corner_radius=4,
        )
        self._live_badge.place(relx=0.02, rely=0.03, anchor="nw")
        self._rec_badge = ctk.CTkLabel(
            self._video_container, text=" REC ", font=ctk.CTkFont(_FONT_FAMILY, size=10),
            fg_color="red", text_color="white", corner_radius=4,
        )
        self._rec_badge.place(relx=0.98, rely=0.03, anchor="ne")
        self._btn_next_cam = ctk.CTkButton(video_row, text="›", width=40, height=40, corner_radius=20, fg_color="#E2E8F0", hover_color="#CBD5E1", font=ctk.CTkFont(_FONT_FAMILY, size=24), command=self._next_camera)
        self._btn_next_cam.pack(side="right", padx=(8, 0), pady=40)
        cam_info = ctk.CTkFrame(left_pane, fg_color="transparent")
        cam_info.pack(fill="x", pady=(8, 0))
        self._camera_title_label = ctk.CTkLabel(cam_info, text="", font=ctk.CTkFont(_FONT_FAMILY, size=14, weight="bold"), text_color=_COLORS["text_primary"])
        self._camera_title_label.pack(anchor="w")
        self._camera_code_label = ctk.CTkLabel(cam_info, text="", font=ctk.CTkFont(_FONT_FAMILY, size=11), text_color=_COLORS["text_secondary"])
        self._camera_code_label.pack(anchor="w")
        cam_actions = ctk.CTkFrame(left_pane, fg_color="transparent")
        cam_actions.pack(fill="x", pady=(4, 0))
        ctk.CTkButton(cam_actions, text="Mở video...", width=100, font=ctk.CTkFont(_FONT_FAMILY), command=self._on_open_video).pack(side="left", padx=(0, 8))
        self._btn_back_camera = ctk.CTkButton(cam_actions, text="Quay lại camera", width=120, fg_color="gray", font=ctk.CTkFont(_FONT_FAMILY), command=self._on_back_to_camera)
        self._btn_back_camera.pack(side="left")
        self._btn_back_camera.pack_forget()

        # Right: Hoạt động toàn cục
        right_pane = ctk.CTkFrame(content, fg_color="transparent", width=320)
        right_pane.pack(side="right", fill="y", padx=(12, 0))
        right_pane.pack_propagate(False)
        list_header = ctk.CTkFrame(right_pane, fg_color="white", corner_radius=10, border_width=1, border_color="#E2E8F0")
        list_header.pack(fill="x", pady=(0, 8))
        inner_header = ctk.CTkFrame(list_header, fg_color="transparent")
        inner_header.pack(fill="x", padx=14, pady=10)
        ctk.CTkLabel(inner_header, text="⚠", font=ctk.CTkFont(_FONT_FAMILY, size=18), text_color=_COLORS["total_accent"]).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(inner_header, text="HOẠT ĐỘNG TOÀN CỤC", font=ctk.CTkFont(_FONT_FAMILY, size=13, weight="bold"), text_color=_COLORS["text_primary"]).pack(side="left")
        self._violation_count_badge = ctk.CTkLabel(
            inner_header, text=" 0 ", font=ctk.CTkFont(_FONT_FAMILY, size=12),
            fg_color=_COLORS["total_accent"], text_color="white", corner_radius=10,
        )
        self._violation_count_badge.pack(side="left", padx=8)
        ctk.CTkButton(
            inner_header, text="Xóa tất cả", fg_color="transparent", text_color=_COLORS["text_secondary"], width=80,
            font=ctk.CTkFont(_FONT_FAMILY), command=self._clear_violations,
        ).pack(side="right")
        self._violation_scroll = ctk.CTkScrollableFrame(right_pane, fg_color="transparent")
        self._violation_scroll.pack(fill="both", expand=True)

        self._build_cameras_page()
        self._build_settings_page()
        self._show_page("dashboard")
        self._update_camera_labels()

    def _show_page(self, name: str):
        self._dashboard_frame.pack_forget()
        self._cameras_frame.pack_forget()
        self._settings_frame.pack_forget()
        if name == "dashboard":
            self._dashboard_frame.pack(fill="both", expand=True)
            self._btn_dashboard.configure(fg_color="#1E293B", text_color="white")
            self._btn_cameras.configure(fg_color="transparent", text_color=_COLORS["text_primary"])
            self._btn_settings.configure(fg_color="transparent", text_color=_COLORS["text_primary"])
        elif name == "cameras":
            self._cameras_frame.pack(fill="both", expand=True)
            self._btn_dashboard.configure(fg_color="transparent", text_color=_COLORS["text_primary"])
            self._btn_cameras.configure(fg_color="#1E293B", text_color="white")
            self._btn_settings.configure(fg_color="transparent", text_color=_COLORS["text_primary"])
        else:
            self._settings_frame.pack(fill="both", expand=True)
            self._btn_dashboard.configure(fg_color="transparent", text_color=_COLORS["text_primary"])
            self._btn_cameras.configure(fg_color="transparent", text_color=_COLORS["text_primary"])
            self._btn_settings.configure(fg_color="#1E293B", text_color="white")

    def _build_cameras_page(self):
        # Header: "Lưới Camera" trái, "X LUỒNG ĐANG HOẠT ĐỘNG" phải
        header = ctk.CTkFrame(self._cameras_frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(
            header, text="Lưới Camera",
            font=ctk.CTkFont(_FONT_FAMILY, size=22, weight="bold"),
            text_color=_COLORS["text_primary"],
        ).pack(side="left")
        n_active = len(self._cameras)
        ctk.CTkLabel(
            header, text=f"{n_active} LUỒNG ĐANG HOẠT ĐỘNG",
            font=ctk.CTkFont(_FONT_FAMILY, size=12),
            text_color=_COLORS["text_secondary"],
        ).pack(side="right")
        # Lưới 2x2 (2 cột) giống ảnh
        grid = ctk.CTkFrame(self._cameras_frame, fg_color="transparent")
        grid.pack(fill="both", expand=True)
        cols = 2
        for c in range(cols):
            grid.columnconfigure(c, weight=1)
        for i, cam in enumerate(self._cameras):
            row, col = i // cols, i % cols
            card = ctk.CTkFrame(grid, fg_color="#1E293B", corner_radius=12, border_width=0)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            grid.rowconfigure(row, weight=1)
            # Preview chiếm full thẻ; badge TRỰC TIẾP góc trên-phải
            preview = ctk.CTkFrame(card, fg_color="#1E293B", corner_radius=12)
            preview.pack(fill="both", expand=True)
            ctk.CTkLabel(
                preview, text=" TRỰC TIẾP ",
                font=ctk.CTkFont(_FONT_FAMILY, size=10),
                fg_color="#16A34A", text_color="white", corner_radius=4,
            ).place(relx=0.98, rely=0.06, anchor="ne")
            # Tên + mã camera overlay góc dưới-trái, chữ trắng
            bottom = ctk.CTkFrame(preview, fg_color="transparent")
            bottom.place(relx=0, rely=1, anchor="sw", relwidth=1, y=-12, x=12)
            ctk.CTkLabel(
                bottom, text=cam.get("name", "Camera"),
                font=ctk.CTkFont(_FONT_FAMILY, size=16, weight="bold"),
                text_color="white",
            ).pack(anchor="w")
            ctk.CTkLabel(
                bottom, text=cam.get("id", "CAM-?"),
                font=ctk.CTkFont(_FONT_FAMILY, size=12),
                text_color="#E2E8F0",
            ).pack(anchor="w")
            def _go(idx):
                self._switch_to_camera_index(idx)
                self._show_page("dashboard")
            card.bind("<Button-1>", lambda e, idx=i: _go(idx))
            for ch in card.winfo_children():
                ch.bind("<Button-1>", lambda e, idx=i: _go(idx))
        # Thẻ "Thêm camera" bên dưới lưới (viền đứt, icon + chữ)
        add_row = len(self._cameras) // cols
        add_col = len(self._cameras) % cols
        grid.rowconfigure(add_row, weight=1)
        add_card = ctk.CTkFrame(
            grid, fg_color="#F1F5F9", corner_radius=12,
            border_width=2, border_color="#94A3B8",
        )
        add_card.grid(row=add_row, column=add_col, padx=8, pady=8, sticky="nsew")
        inner = ctk.CTkFrame(add_card, fg_color="transparent")
        inner.pack(fill="both", expand=True)
        ctk.CTkLabel(
            inner, text="➕",
            font=ctk.CTkFont(_FONT_FAMILY, size=40),
            text_color=_COLORS["text_secondary"],
        ).pack(pady=(20, 8))
        ctk.CTkLabel(
            inner, text="Thêm camera",
            font=ctk.CTkFont(_FONT_FAMILY, size=14, weight="bold"),
            text_color=_COLORS["text_primary"],
        ).pack()
        ctk.CTkButton(inner, text="Thêm", width=100, command=self._on_add_camera).pack(pady=(12, 20))

    def _build_settings_page(self):
        """Màn Cài đặt full-page: Độ nhạy nhận diện, Lưu trữ & hệ thống, Camera & Model."""
        main = ctk.CTkFrame(self._settings_frame, fg_color="transparent")
        main.pack(fill="both", expand=True)
        scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ---- Tiêu đề ----
        ctk.CTkLabel(
            scroll, text="Cài đặt hệ thống",
            font=ctk.CTkFont(_FONT_FAMILY, size=22, weight="bold"),
            text_color=_COLORS["text_primary"],
        ).pack(anchor="w", pady=(0, 20))

        def _card(title: str):
            wrap = ctk.CTkFrame(scroll, fg_color="transparent")
            wrap.pack(fill="x", pady=(0, 16))
            ctk.CTkLabel(
                wrap, text=title,
                font=ctk.CTkFont(_FONT_FAMILY, size=12, weight="bold"),
                text_color=_COLORS["text_secondary"],
            ).pack(anchor="w", pady=(0, 8))
            card = ctk.CTkFrame(
                wrap, fg_color="white", corner_radius=12,
                border_width=1, border_color="#E2E8F0",
            )
            card.pack(fill="x")
            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="x", padx=20, pady=16)
            return inner

        # ---- Section 1: ĐỘ NHẠY NHẬN DIỆN ----
        s1 = _card("ĐỘ NHẠY NHẬN DIỆN")
        ctk.CTkLabel(s1, text="Ngưỡng tin cậy", font=ctk.CTkFont(_FONT_FAMILY, size=12), text_color=_COLORS["text_primary"]).pack(anchor="w", pady=(0, 6))
        row_slider = ctk.CTkFrame(s1, fg_color="transparent")
        row_slider.pack(fill="x", pady=(0, 16))
        self._settings_confidence_label = ctk.CTkLabel(
            row_slider, text="85%",
            font=ctk.CTkFont(_FONT_FAMILY, size=12),
            text_color=_COLORS["text_secondary"],
        )
        self._settings_confidence_label.pack(side="right", padx=(0, 0))
        self._settings_confidence_slider = ctk.CTkSlider(
            row_slider, from_=10, to=95, number_of_steps=85,
            width=300, height=16, command=self._on_settings_confidence_slider,
        )
        self._settings_confidence_slider.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self._settings_confidence_slider.set(int(CONFIDENCE_THRESHOLD * 100))

        def _toggle_row(parent, icon_text: str, label: str, initial_on: bool = False):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=(0, 10))
            ctk.CTkLabel(row, text=icon_text, font=ctk.CTkFont(_FONT_FAMILY, size=16), text_color=_COLORS["text_primary"]).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(_FONT_FAMILY, size=13), text_color=_COLORS["text_primary"]).pack(side="left", fill="x", expand=True)
            sw = ctk.CTkSwitch(row, text="", width=40)
            sw.pack(side="right")
            if initial_on:
                sw.select()
            else:
                sw.deselect()
            return sw

        _toggle_row(s1, "🛡", "Nhận diện mũ", initial_on=True)
        _toggle_row(s1, "⚡", "Tracking (ByteTrack)", initial_on=True)

        # ---- Section 2: LƯU TRỮ & HỆ THỐNG (giống ảnh) ----
        s2 = _card("LƯU TRỮ & HỆ THỐNG")
        row_storage = ctk.CTkFrame(s2, fg_color="transparent")
        row_storage.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(row_storage, text="📦", font=ctk.CTkFont(_FONT_FAMILY, size=16)).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(row_storage, text="Dung lượng đám mây", font=ctk.CTkFont(_FONT_FAMILY, size=12, weight="bold"), text_color=_COLORS["text_primary"]).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row_storage, text="QUẢN LÝ", width=90, height=28, fg_color="#64748B", hover_color="#475569", font=ctk.CTkFont(_FONT_FAMILY, size=11)).pack(side="right")
        ctk.CTkLabel(s2, text="1.2 TB / 2.0 TB đã dùng", font=ctk.CTkFont(_FONT_FAMILY, size=11), text_color=_COLORS["text_secondary"]).pack(anchor="w", pady=(0, 6))
        progress = ctk.CTkProgressBar(s2, width=400, height=8, progress_color="#2563EB", fg_color="#E2E8F0")
        progress.pack(fill="x", pady=(0, 16))
        progress.set(0.6)
        row_backup = ctk.CTkFrame(s2, fg_color="transparent")
        row_backup.pack(fill="x")
        ctk.CTkLabel(row_backup, text="🔄", font=ctk.CTkFont(_FONT_FAMILY, size=16)).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(row_backup, text="Tự động sao lưu", font=ctk.CTkFont(_FONT_FAMILY, size=12, weight="bold"), text_color=_COLORS["text_primary"]).pack(side="left", fill="x", expand=True)
        ctk.CTkSwitch(row_backup, text="").pack(side="right")
        ctk.CTkLabel(s2, text="Hàng ngày lúc 02:00 AM", font=ctk.CTkFont(_FONT_FAMILY, size=11), text_color=_COLORS["text_secondary"]).pack(anchor="w", pady=(4, 0))

        # ---- Section 3: Camera & Model ----
        s3 = _card("CAMERA & MODEL")
        ctk.CTkLabel(s3, text="Index camera", font=ctk.CTkFont(_FONT_FAMILY, size=12), text_color=_COLORS["text_primary"]).pack(anchor="w", pady=(0, 4))
        self._settings_camera_index = ctk.CTkEntry(s3, placeholder_text="0", height=36, width=320)
        self._settings_camera_index.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(s3, text="Tên khu vực", font=ctk.CTkFont(_FONT_FAMILY, size=12), text_color=_COLORS["text_primary"]).pack(anchor="w", pady=(0, 4))
        self._settings_area_name = ctk.CTkEntry(s3, placeholder_text="Khu vực A", height=36, width=320)
        self._settings_area_name.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(s3, text="Throttle (giây)", font=ctk.CTkFont(_FONT_FAMILY, size=12), text_color=_COLORS["text_primary"]).pack(anchor="w", pady=(0, 4))
        self._settings_throttle = ctk.CTkEntry(s3, placeholder_text="0.8", height=36, width=320)
        self._settings_throttle.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(s3, text="Đường dẫn model .pt", font=ctk.CTkFont(_FONT_FAMILY, size=12), text_color=_COLORS["text_primary"]).pack(anchor="w", pady=(0, 4))
        model_row = ctk.CTkFrame(s3, fg_color="transparent")
        model_row.pack(fill="x", pady=(0, 4))
        self._settings_model_path = ctk.CTkEntry(model_row, placeholder_text="helmet_best.pt", height=36)
        self._settings_model_path.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            model_row,
            text="Chọn file...",
            width=110,
            height=36,
            fg_color="#64748B",
            hover_color="#475569",
            font=ctk.CTkFont(_FONT_FAMILY, size=11, weight="bold"),
            command=self._pick_model_file_for_settings_page,
        ).pack(side="right", padx=(10, 0))
        self._settings_confidence_label.configure(text=f"{int(CONFIDENCE_THRESHOLD * 100)}%")
        self._settings_camera_index.insert(0, str(CAMERA_INDEX))
        self._settings_area_name.insert(0, CAMERA_AREA_NAME)
        self._settings_throttle.insert(0, str(VIOLATION_THROTTLE_SECONDS))
        if MODEL_PATH and os.path.isabs(MODEL_PATH):
            self._settings_model_path.insert(0, MODEL_PATH)
        else:
            self._settings_model_path.insert(0, os.path.basename(MODEL_PATH) if MODEL_PATH else "")

        # ---- Footer: nút Lưu cài đặt ----
        sep = ctk.CTkFrame(main, fg_color="#E2E8F0", height=1)
        sep.pack(fill="x", pady=(16, 12))
        btn_frame = ctk.CTkFrame(main, fg_color="transparent")
        btn_frame.pack(fill="x")
        ctk.CTkButton(
            btn_frame, text="Lưu cài đặt",
            width=140, height=40,
            fg_color="#2563EB", hover_color="#1D4ED8",
            font=ctk.CTkFont(_FONT_FAMILY, size=13, weight="bold"),
            command=self._save_settings_page,
        ).pack(side="right")

    def _on_settings_confidence_slider(self, value):
        if hasattr(self, "_settings_confidence_label") and self._settings_confidence_label.winfo_exists():
            self._settings_confidence_label.configure(text=f"{int(value)}%")

    def _pick_model_file_for_settings_page(self):
        from tkinter import filedialog

        selected_path = filedialog.askopenfilename(
            title="Chọn file model (.pt)",
            filetypes=[
                ("PyTorch model", "*.pt"),
                ("Tất cả", "*.*"),
            ],
        )
        if not selected_path:
            return
        if not hasattr(self, "_settings_model_path") or not self._settings_model_path.winfo_exists():
            return
        self._settings_model_path.delete(0, "end")
        self._settings_model_path.insert(0, selected_path)

    def _save_settings_page(self):
        """Lưu từ màn Cài đặt (ghi settings.json, reload, áp dụng camera)."""
        try:
            camera_index = int((self._settings_camera_index.get() or "").strip() or "0")
        except ValueError:
            camera_index = 0
        area_name = (self._settings_area_name.get() or "").strip() or CAMERA_AREA_NAME
        area_name = area_name[:80]
        try:
            confidence = float(self._settings_confidence_slider.get()) / 100.0
            confidence = max(0.1, min(0.95, confidence))
        except Exception:
            confidence = CONFIDENCE_THRESHOLD
        try:
            throttle = float((self._settings_throttle.get() or "").strip() or "0.8")
            throttle = max(0.2, min(10.0, throttle))
        except ValueError:
            throttle = VIOLATION_THROTTLE_SECONDS
        model_path = (self._settings_model_path.get() or "").strip()
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)
        data = {
            "camera_index": camera_index,
            "camera_area_name": area_name,
            "confidence_threshold": confidence,
            "violation_throttle_seconds": throttle,
            "model_path": model_path or None,
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(USER_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        load_user_settings()
        if self._camera_thread:
            self._camera_thread.stop()
        self._cameras = load_cameras()
        self._is_playing_video_file = False
        self._apply_current_camera()
        if hasattr(self, "_btn_back_camera") and self._btn_back_camera.winfo_ismapped():
            self._btn_back_camera.pack_forget()

    def _prev_camera(self):
        if self._is_playing_video_file or not self._cameras:
            return
        self._current_camera_index = (self._current_camera_index - 1) % len(self._cameras)
        self._apply_current_camera()

    def _next_camera(self):
        if self._is_playing_video_file or not self._cameras:
            return
        self._current_camera_index = (self._current_camera_index + 1) % len(self._cameras)
        self._apply_current_camera()

    def _switch_to_camera_index(self, index: int):
        if not self._cameras:
            return
        self._current_camera_index = max(0, min(index, len(self._cameras) - 1))
        self._is_playing_video_file = False
        self._apply_current_camera()

    def _apply_current_camera(self):
        if self._camera_thread:
            self._camera_thread.stop()
        self._current_source = self._cameras[self._current_camera_index].get("source", 0) if self._cameras else CAMERA_INDEX
        self._start_source()
        self._update_camera_labels()

    def _current_camera_name(self) -> str:
        if self._is_playing_video_file:
            return "Video"
        if self._cameras and 0 <= self._current_camera_index < len(self._cameras):
            return self._cameras[self._current_camera_index].get("name", "Camera")
        return CAMERA_AREA_NAME

    def _update_camera_labels(self):
        if self._is_playing_video_file:
            return
        if self._cameras and 0 <= self._current_camera_index < len(self._cameras):
            c = self._cameras[self._current_camera_index]
            self._camera_title_label.configure(text=c.get("name", "Camera"))
            self._camera_code_label.configure(text=f"MÃ CAMERA: {c.get('id', 'CAM-?')} • Live")
        else:
            self._camera_title_label.configure(text=CAMERA_AREA_NAME)
            self._camera_code_label.configure(text="MÃ CAMERA: CAM-1 • Live")

    def _on_add_camera(self):
        d = ctk.CTkToplevel(self.root)
        d.title("Thêm camera IP")
        d.geometry("400x260")
        d.transient(self.root)
        f = ctk.CTkFrame(d, fg_color="transparent")
        f.pack(fill="both", expand=True, padx=20, pady=20)
        ctk.CTkLabel(f, text="Tên khu vực", font=ctk.CTkFont(_FONT_FAMILY, size=12)).pack(anchor="w")
        name_entry = ctk.CTkEntry(f, placeholder_text="Ví dụ: Khu vực C", width=320)
        name_entry.pack(fill="x", pady=(4, 12))
        ctk.CTkLabel(f, text="Mã camera (CAM-x)", font=ctk.CTkFont(_FONT_FAMILY, size=12)).pack(anchor="w")
        id_entry = ctk.CTkEntry(f, placeholder_text="CAM-5", width=320)
        id_entry.pack(fill="x", pady=(4, 12))
        ctk.CTkLabel(f, text="Nguồn (số index 0,1,2... hoặc URL RTSP)", font=ctk.CTkFont(_FONT_FAMILY, size=12)).pack(anchor="w")
        source_entry = ctk.CTkEntry(f, placeholder_text="0 hoặc rtsp://...", width=320)
        source_entry.pack(fill="x", pady=(4, 16))
        def do_add():
            name = (name_entry.get() or "").strip() or "Camera mới"
            id_val = (id_entry.get() or "").strip() or f"CAM-{len(self._cameras)+1}"
            src = source_entry.get().strip()
            try:
                src = int(src) if src and src.isdigit() else src
            except ValueError:
                src = 0
            self._cameras.append({"name": name, "id": id_val, "source": src})
            save_cameras(self._cameras)
            d.destroy()
            self._cameras_frame.destroy()
            self._cameras_frame = ctk.CTkFrame(self._content, fg_color="transparent")
            self._build_cameras_page()
            self._cameras_frame.pack_forget()
            self._dashboard_frame.pack(fill="both", expand=True)
        ctk.CTkButton(f, text="Thêm", width=100, command=do_add).pack(anchor="w")

    def _on_settings(self):
        def on_saved():
            from src.config.settings import load_cameras as reload_cameras
            if self._camera_thread:
                self._camera_thread.stop()
            self._cameras = reload_cameras()
            self._is_playing_video_file = False
            self._apply_current_camera()
            self._btn_back_camera.pack_forget()
        SettingsDialog(self.root, on_saved=on_saved).focus_set()

    def _on_open_video(self):
        """Chọn file video và chạy nhận diện trên video."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Chọn video để test nhận diện",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("MP4", "*.mp4"),
                ("Tất cả", "*.*"),
            ],
        )
        if not path:
            return
        if self._camera_thread:
            self._camera_thread.stop()
        self._current_source = path
        self._is_playing_video_file = True
        self._start_source()
        import os
        self._camera_title_label.configure(text=f"Video: {os.path.basename(path)}")
        self._camera_code_label.configure(text="Phát file video")
        self._btn_back_camera.pack(side="left", padx=(0, 8))
        self._video_label.configure(text="")

    def _on_back_to_camera(self):
        self._is_playing_video_file = False
        if self._camera_thread:
            self._camera_thread.stop()
        self._apply_current_camera()
        self._btn_back_camera.pack_forget()
        self._video_label.configure(text="")

    def _on_video_ended(self):
        """Gọi khi video file phát xong."""
        if self._video_label:
            self.root.after(0, lambda: self._video_label.configure(text="Video kết thúc."))

    def _clear_violations(self):
        self._violations.clear()
        self._refresh_violation_list()
        self._update_kpis(0, 0, 0, 0)
        self._update_alert(0)

    def _start_camera(self):
        """Khởi động lại camera (gọi từ on_saved trong Cài đặt)."""
        from src.config.settings import CAMERA_INDEX
        self._current_source = CAMERA_INDEX
        self._start_source()

    def _start_source(self):
        """Khởi động nguồn hiện tại (camera hoặc file video)."""
        self._camera_thread = CameraThread(
            on_frame=self._on_frame,
            source=self._current_source,
            on_video_end=self._on_video_ended,
        )
        self._camera_thread.start()

    def _on_frame(self, result: DetectionResult):
        """Callback từ camera thread: violations đã dedup theo person_id trong CameraThread."""
        to_add = list(result.violations)
        self.root.after(0, lambda: self._on_frame_ui(result, to_add))

    def _on_frame_ui(self, result: DetectionResult, to_add: List[Violation]):
        """Chạy trên main thread: cập nhật ảnh, thêm vi phạm, refresh UI."""
        loc = self._current_camera_name()
        for v in to_add:
            self._violations.append(replace(v, location=loc))
        # Cập nhật ảnh video
        frame = result.frame
        if frame is not None and frame.size > 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            max_w, max_h = 640, 480
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(frame_rgb).resize((new_w, new_h), PILImage.Resampling.LANCZOS)
                self._set_video_image(pil_img, new_w, new_h)
        total = len(self._violations)
        h_count = sum(1 for x in self._violations if x.violation_type == ViolationType.MISSING_HELMET)
        v_count = sum(1 for x in self._violations if x.violation_type == ViolationType.MISSING_VEST)
        b_count = sum(1 for x in self._violations if x.violation_type == ViolationType.MISSING_BOTH)
        self._update_kpis(total, h_count, v_count, b_count)
        if to_add:
            self._refresh_violation_list()
        self._update_alert(total)

    def _set_video_image(self, pil_img: "Image.Image", w: int, h: int):
        if self._video_label is None:
            return
        # CTkImage yêu cầu light_image và dark_image cùng kích thước; khi đổi nguồn (camera ↔ video) kích thước frame có thể đổi → tạo mới CTkImage thay vì configure.
        current_size = getattr(self, "_video_image_size", None)
        if self._video_image is None or current_size != (w, h):
            self._video_image_size = (w, h)
            self._video_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
            self._video_label.configure(image=self._video_image, text="")
        else:
            self._video_image.configure(light_image=pil_img, dark_image=pil_img, size=(w, h))
            self._video_label.configure(image=self._video_image, text="")

    def _update_kpis(self, total: int, helmet: int, vest: int, both: int):
        self._kpi_total.set_value(total)
        self._kpi_helmet.set_value(helmet)
        self._kpi_vest.set_value(vest)
        self._kpi_both.set_value(both)

    def _update_alert(self, total: int):
        if total > 0:
            self._alert_bar_placeholder.pack_forget()
            self._alert_bar.pack(fill="x", padx=24, pady=(0, 16))
            self._alert_label.configure(
                text=f"Cảnh báo! Phát hiện {total} vi phạm an toàn lao động. Vui lòng kiểm tra và xử lý ngay lập tức."
            )
        else:
            self._alert_bar.pack_forget()
            self._alert_bar_placeholder.pack(fill="x")

    def _refresh_violation_list(self):
        for w in self._violation_scroll.winfo_children():
            w.destroy()
        for v in reversed(list(self._violations)):
            card = ViolationCard(self._violation_scroll, v)
            card.pack(fill="x", pady=4)
        self._violation_count_badge.configure(text=f" {len(self._violations)} ")

    def run(self):
        try:
            self.root.mainloop()
        finally:
            if self._camera_thread:
                self._camera_thread.stop()
