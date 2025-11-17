"""
随机生成素材等对话框（参考 material_editor/ui/dialogs.py 精简版）
"""

from __future__ import annotations

import random
from typing import List, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def numpy_bgr_to_qimage(img_bgr: np.ndarray) -> QImage:
    """将 BGR numpy 图像转换为 QImage（RGB888）。"""
    if img_bgr is None:
        return None
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("numpy_bgr_to_qimage 仅支持 BGR 三通道图像")
    h, w, _ = img_bgr.shape
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return qimg.copy()


class VisualRangeSelector(QWidget):
    """可视化范围选择器（精简版）"""

    range_changed = Signal(int, int, int, int)  # x_min, y_min, x_max, y_max

    def __init__(
        self, canvas_width: int, canvas_height: int, background_image=None, parent=None
    ):
        super().__init__(parent)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.preview_width = 250
        self.preview_height = 140

        self.background_image = background_image
        self.background_pixmap = None
        if self.background_image is not None:
            self._create_background_preview()

        # 初始范围 10%~90%
        self.x_min_ratio = 0.1
        self.y_min_ratio = 0.1
        self.x_max_ratio = 0.9
        self.y_max_ratio = 0.9

        self.dragging = False
        self.drag_corner = (
            None  # 'top-left' / 'top-right' / 'bottom-left' / 'bottom-right' / 'move'
        )
        self.last_mouse_pos = None

        self.setFixedSize(self.preview_width, self.preview_height)
        self.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")

    def _create_background_preview(self):
        try:
            resized_img = cv2.resize(
                self.background_image,
                (self.preview_width, self.preview_height),
                interpolation=cv2.INTER_AREA,
            )
            qimage = numpy_bgr_to_qimage(resized_img)
            if qimage:
                self.background_pixmap = QPixmap.fromImage(qimage)
        except Exception as e:
            print(f"创建背景预览失败: {e}")
            self.background_pixmap = None

    def set_range(self, x_min: int, y_min: int, x_max: int, y_max: int):
        self.x_min_ratio = x_min / self.canvas_width
        self.y_min_ratio = y_min / self.canvas_height
        self.x_max_ratio = x_max / self.canvas_width
        self.y_max_ratio = y_max / self.canvas_height
        self.update()

    def get_range(self) -> Tuple[int, int, int, int]:
        x_min = int(self.x_min_ratio * self.canvas_width)
        y_min = int(self.y_min_ratio * self.canvas_height)
        x_max = int(self.x_max_ratio * self.canvas_width)
        y_max = int(self.y_max_ratio * self.canvas_height)
        return x_min, y_min, x_max, y_max

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景
        if self.background_pixmap:
            painter.drawPixmap(
                0, 0, self.preview_width, self.preview_height, self.background_pixmap
            )
        else:
            painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.PenStyle.DotLine))
            for i in range(0, self.preview_width, 20):
                painter.drawLine(i, 0, i, self.preview_height)
            for i in range(0, self.preview_height, 20):
                painter.drawLine(0, i, self.preview_width, i)

        x1 = int(self.x_min_ratio * self.preview_width)
        y1 = int(self.y_min_ratio * self.preview_height)
        x2 = int(self.x_max_ratio * self.preview_width)
        y2 = int(self.y_max_ratio * self.preview_height)

        # 未选区域半透明
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(QColor(128, 128, 128, 150)))
        if y1 > 0:
            painter.drawRect(0, 0, self.preview_width, y1)
        if y2 < self.preview_height:
            painter.drawRect(0, y2, self.preview_width, self.preview_height - y2)
        if x1 > 0:
            painter.drawRect(0, y1, x1, y2 - y1)
        if x2 < self.preview_width:
            painter.drawRect(x2, y1, self.preview_width - x2, y2 - y1)

        # 选框
        painter.setPen(QPen(QColor(0, 120, 215), 2))
        painter.setBrush(QBrush())
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        # 控制点
        handle_size = 6
        painter.setPen(QPen(QColor(0, 120, 215), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawRect(
            x1 - handle_size // 2, y1 - handle_size // 2, handle_size, handle_size
        )
        painter.drawRect(
            x2 - handle_size // 2, y1 - handle_size // 2, handle_size, handle_size
        )
        painter.drawRect(
            x1 - handle_size // 2, y2 - handle_size // 2, handle_size, handle_size
        )
        painter.drawRect(
            x2 - handle_size // 2, y2 - handle_size // 2, handle_size, handle_size
        )

        # 尺寸标签
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))

        top_text = f"{int(self.x_min_ratio * self.canvas_width)}, {int(self.y_min_ratio * self.canvas_height)}"
        text_rect1 = painter.fontMetrics().boundingRect(top_text)
        text_rect1.moveTopLeft(QPoint(x1 + 2, y1 + 2))
        painter.drawRect(text_rect1.adjusted(-2, -1, 2, 1))
        painter.drawText(text_rect1, Qt.AlignmentFlag.AlignLeft, top_text)

        bottom_text = f"{int(self.x_max_ratio * self.canvas_width)}, {int(self.y_max_ratio * self.canvas_height)}"
        text_rect2 = painter.fontMetrics().boundingRect(bottom_text)
        text_rect2.moveBottomRight(QPoint(x2 - 2, y2 - 2))
        painter.drawRect(text_rect2.adjusted(-2, -1, 2, 1))
        painter.drawText(text_rect2, Qt.AlignmentFlag.AlignRight, bottom_text)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        x, y = pos.x(), pos.y()
        x1 = int(self.x_min_ratio * self.preview_width)
        y1 = int(self.y_min_ratio * self.preview_height)
        x2 = int(self.x_max_ratio * self.preview_width)
        y2 = int(self.y_max_ratio * self.preview_height)
        handle_size = 8

        if abs(x - x1) <= handle_size and abs(y - y1) <= handle_size:
            self.drag_corner = "top-left"
        elif abs(x - x2) <= handle_size and abs(y - y1) <= handle_size:
            self.drag_corner = "top-right"
        elif abs(x - x1) <= handle_size and abs(y - y2) <= handle_size:
            self.drag_corner = "bottom-left"
        elif abs(x - x2) <= handle_size and abs(y - y2) <= handle_size:
            self.drag_corner = "bottom-right"
        elif x1 <= x <= x2 and y1 <= y <= y2:
            self.drag_corner = "move"
        else:
            return

        self.dragging = True
        self.last_mouse_pos = pos
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not (self.dragging and self.last_mouse_pos):
            return
        pos = event.position().toPoint()
        dx = pos.x() - self.last_mouse_pos.x()
        dy = pos.y() - self.last_mouse_pos.y()
        dx_ratio = dx / self.preview_width
        dy_ratio = dy / self.preview_height

        if self.drag_corner == "top-left":
            self.x_min_ratio = max(
                0, min(self.x_max_ratio - 0.05, self.x_min_ratio + dx_ratio)
            )
            self.y_min_ratio = max(
                0, min(self.y_max_ratio - 0.05, self.y_min_ratio + dy_ratio)
            )
        elif self.drag_corner == "top-right":
            self.x_max_ratio = min(
                1, max(self.x_min_ratio + 0.05, self.x_max_ratio + dx_ratio)
            )
            self.y_min_ratio = max(
                0, min(self.y_max_ratio - 0.05, self.y_min_ratio + dy_ratio)
            )
        elif self.drag_corner == "bottom-left":
            self.x_min_ratio = max(
                0, min(self.x_max_ratio - 0.05, self.x_min_ratio + dx_ratio)
            )
            self.y_max_ratio = min(
                1, max(self.y_min_ratio + 0.05, self.y_max_ratio + dy_ratio)
            )
        elif self.drag_corner == "bottom-right":
            self.x_max_ratio = min(
                1, max(self.x_min_ratio + 0.05, self.x_max_ratio + dx_ratio)
            )
            self.y_max_ratio = min(
                1, max(self.y_min_ratio + 0.05, self.y_max_ratio + dy_ratio)
            )
        elif self.drag_corner == "move":
            width_ratio = self.x_max_ratio - self.x_min_ratio
            height_ratio = self.y_max_ratio - self.y_min_ratio
            new_x_min = self.x_min_ratio + dx_ratio
            new_y_min = self.y_min_ratio + dy_ratio
            if new_x_min < 0:
                new_x_min = 0
            elif new_x_min + width_ratio > 1:
                new_x_min = 1 - width_ratio
            if new_y_min < 0:
                new_y_min = 0
            elif new_y_min + height_ratio > 1:
                new_y_min = 1 - height_ratio
            self.x_min_ratio = new_x_min
            self.y_min_ratio = new_y_min
            self.x_max_ratio = new_x_min + width_ratio
            self.y_max_ratio = new_y_min + height_ratio

        self.last_mouse_pos = pos
        self.update()
        x_min, y_min, x_max, y_max = self.get_range()
        self.range_changed.emit(x_min, y_min, x_max, y_max)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.drag_corner = None
            self.last_mouse_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)


class RandomGenerateDialog(QDialog):
    """随机生成素材的配置对话框（精简版）"""

    def __init__(
        self,
        material_names: List[str],
        canvas_size: Tuple[int, int],
        background_image=None,
        parent=None,
    ):
        super().__init__(parent)
        self.material_names = material_names
        self.canvas_width, self.canvas_height = canvas_size
        self.background_image = background_image

        self.setWindowTitle("随机生成素材")
        self.setModal(True)
        self.resize(650, 780)

        self._init_ui()

    # --- UI 构建 ---
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        main_layout.addWidget(scroll_area)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)

        # 基本设置
        basic_group = QGroupBox("基本设置")
        basic_layout = QFormLayout(basic_group)
        basic_layout.setSpacing(8)
        basic_layout.setHorizontalSpacing(15)

        count_mode_layout = QHBoxLayout()
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 200)
        self.count_spin.setValue(20)
        self.count_spin.setMinimumWidth(80)
        count_mode_layout.addWidget(QLabel("数量:"))
        count_mode_layout.addWidget(self.count_spin)
        count_mode_layout.addSpacing(20)
        count_mode_layout.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["随机选择素材", "使用所有素材", "仅使用第一个素材", "均匀分布所有素材"]
        )
        count_mode_layout.addWidget(self.mode_combo)
        count_mode_layout.addStretch()
        basic_layout.addRow("生成设置:", count_mode_layout)
        layout.addWidget(basic_group)

        # 位置设置
        position_group = QGroupBox("位置设置")
        position_layout = QVBoxLayout(position_group)

        visual_section = QVBoxLayout()
        range_header = QHBoxLayout()
        range_header.addWidget(QLabel("生成范围预览:"))
        range_header.addStretch()
        reset_range_btn = QPushButton("重置范围")
        reset_range_btn.clicked.connect(self._reset_range)
        range_header.addWidget(reset_range_btn)
        visual_section.addLayout(range_header)

        self.range_selector = VisualRangeSelector(
            self.canvas_width, self.canvas_height, self.background_image
        )
        self.range_selector.range_changed.connect(self._on_visual_range_changed)
        visual_section.addWidget(self.range_selector, 0, Qt.AlignmentFlag.AlignCenter)

        position_layout.addLayout(visual_section)

        controls_section = QVBoxLayout()
        x_range_layout = QHBoxLayout()
        x_range_layout.addWidget(QLabel("X范围:"))
        self.x_min_spin = QSpinBox()
        self.x_min_spin.setRange(0, self.canvas_width)
        self.x_min_spin.setValue(int(self.canvas_width * 0.1))
        self.x_min_spin.setMinimumWidth(100)
        self.x_min_spin.valueChanged.connect(self._on_numeric_range_changed)
        x_range_layout.addWidget(self.x_min_spin)
        x_range_layout.addWidget(QLabel("到"))
        self.x_max_spin = QSpinBox()
        self.x_max_spin.setRange(0, self.canvas_width)
        self.x_max_spin.setValue(int(self.canvas_width * 0.9))
        self.x_max_spin.setMinimumWidth(100)
        self.x_max_spin.valueChanged.connect(self._on_numeric_range_changed)
        x_range_layout.addWidget(self.x_max_spin)
        x_range_layout.addStretch()
        controls_section.addLayout(x_range_layout)

        y_range_layout = QHBoxLayout()
        y_range_layout.addWidget(QLabel("Y范围:"))
        self.y_min_spin = QSpinBox()
        self.y_min_spin.setRange(0, self.canvas_height)
        self.y_min_spin.setValue(int(self.canvas_height * 0.1))
        self.y_min_spin.setMinimumWidth(100)
        self.y_min_spin.valueChanged.connect(self._on_numeric_range_changed)
        y_range_layout.addWidget(self.y_min_spin)
        y_range_layout.addWidget(QLabel("到"))
        self.y_max_spin = QSpinBox()
        self.y_max_spin.setRange(0, self.canvas_height)
        self.y_max_spin.setValue(int(self.canvas_height * 0.9))
        self.y_max_spin.setMinimumWidth(100)
        self.y_max_spin.valueChanged.connect(self._on_numeric_range_changed)
        y_range_layout.addWidget(self.y_max_spin)
        y_range_layout.addStretch()
        controls_section.addLayout(y_range_layout)

        self.avoid_overlap_check = QCheckBox("尝试避免重叠（简单策略）")
        self.avoid_overlap_check.setChecked(True)
        controls_section.addWidget(self.avoid_overlap_check)

        position_layout.addLayout(controls_section)
        layout.addWidget(position_group)

        # 变换设置
        transform_group = QGroupBox("变换设置")
        transform_layout = QFormLayout(transform_group)

        rotation_layout = QHBoxLayout()
        self.enable_rotation_check = QCheckBox("启用随机旋转")
        self.enable_rotation_check.setChecked(True)
        self.enable_rotation_check.toggled.connect(self._on_rotation_toggled)
        rotation_layout.addWidget(self.enable_rotation_check)
        rotation_layout.addSpacing(20)
        rotation_layout.addWidget(QLabel("范围:"))
        self.rotation_min_spin = QSpinBox()
        self.rotation_min_spin.setRange(-360, 360)
        self.rotation_min_spin.setValue(0)
        rotation_layout.addWidget(self.rotation_min_spin)
        rotation_layout.addWidget(QLabel("到"))
        self.rotation_max_spin = QSpinBox()
        self.rotation_max_spin.setRange(-360, 360)
        self.rotation_max_spin.setValue(360)
        rotation_layout.addWidget(self.rotation_max_spin)
        rotation_layout.addWidget(QLabel("度"))
        rotation_layout.addStretch()
        transform_layout.addRow("旋转:", rotation_layout)

        scale_layout = QHBoxLayout()
        self.enable_scale_check = QCheckBox("启用随机缩放")
        self.enable_scale_check.setChecked(True)
        self.enable_scale_check.toggled.connect(self._on_scale_toggled)
        scale_layout.addWidget(self.enable_scale_check)
        scale_layout.addSpacing(20)
        scale_layout.addWidget(QLabel("范围:"))
        self.scale_min_spin = QSpinBox()
        self.scale_min_spin.setRange(10, 500)
        self.scale_min_spin.setValue(50)
        self.scale_min_spin.setSuffix("%")
        scale_layout.addWidget(self.scale_min_spin)
        scale_layout.addWidget(QLabel("到"))
        self.scale_max_spin = QSpinBox()
        self.scale_max_spin.setRange(10, 500)
        self.scale_max_spin.setValue(150)
        self.scale_max_spin.setSuffix("%")
        scale_layout.addWidget(self.scale_max_spin)
        scale_layout.addStretch()
        transform_layout.addRow("缩放:", scale_layout)
        layout.addWidget(transform_group)

        # 混合模式设置
        blend_group = QGroupBox("混合模式设置")
        blend_layout = QFormLayout(blend_group)
        self.enable_blend_check = QCheckBox("启用随机混合模式")
        self.enable_blend_check.setChecked(True)
        self.enable_blend_check.toggled.connect(self._on_blend_toggled)
        blend_layout.addRow("", self.enable_blend_check)
        blend_modes_layout = QHBoxLayout()
        self.blend_normal_check = QCheckBox("普通")
        self.blend_normal_check.setChecked(True)
        blend_modes_layout.addWidget(self.blend_normal_check)
        self.blend_poisson_normal_check = QCheckBox("泊松融合(正常)")
        self.blend_poisson_normal_check.setChecked(True)
        blend_modes_layout.addWidget(self.blend_poisson_normal_check)
        self.blend_poisson_mixed_check = QCheckBox("泊松融合(混合)")
        self.blend_poisson_mixed_check.setChecked(False)
        blend_modes_layout.addWidget(self.blend_poisson_mixed_check)
        blend_modes_layout.addStretch()
        blend_layout.addRow("可选模式:", blend_modes_layout)
        layout.addWidget(blend_group)

        # 色彩叠加
        color_group = QGroupBox("色彩叠加设置")
        color_layout = QFormLayout(color_group)
        self.enable_color_check = QCheckBox("启用随机色彩叠加")
        self.enable_color_check.setChecked(False)
        self.enable_color_check.toggled.connect(self._on_color_toggled)
        color_layout.addRow("", self.enable_color_check)

        color_mode_layout = QVBoxLayout()
        self.color_random_check = QCheckBox("随机颜色")
        self.color_random_check.setChecked(True)
        color_mode_layout.addWidget(self.color_random_check)
        self.color_background_check = QCheckBox("使用背景颜色（基于生成位置）")
        self.color_background_check.setChecked(False)
        color_mode_layout.addWidget(self.color_background_check)
        self.color_preset_check = QCheckBox("使用预设颜色组")
        self.color_preset_check.setChecked(False)
        color_mode_layout.addWidget(self.color_preset_check)
        self.color_preset_combo = QComboBox()
        self.color_preset_combo.addItems(
            [
                "暖色调",
                "冷色调",
                "大地色",
                "彩虹色",
                "单色调-红",
                "单色调-蓝",
                "单色调-绿",
            ]
        )
        color_mode_layout.addWidget(self.color_preset_combo)
        color_mode_layout.addStretch()
        color_layout.addRow("颜色模式:", color_mode_layout)

        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("透明度范围:"))
        self.color_opacity_min_spinbox = QSpinBox()
        self.color_opacity_min_spinbox.setRange(10, 100)
        self.color_opacity_min_spinbox.setValue(20)
        self.color_opacity_min_spinbox.setSuffix("%")
        opacity_layout.addWidget(self.color_opacity_min_spinbox)
        opacity_layout.addWidget(QLabel("到"))
        self.color_opacity_max_spinbox = QSpinBox()
        self.color_opacity_max_spinbox.setRange(10, 100)
        self.color_opacity_max_spinbox.setValue(60)
        self.color_opacity_max_spinbox.setSuffix("%")
        opacity_layout.addWidget(self.color_opacity_max_spinbox)
        opacity_layout.addStretch()
        color_layout.addRow("", opacity_layout)
        layout.addWidget(color_group)

        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        generate_btn = QPushButton("生成")
        generate_btn.clicked.connect(self.accept)
        button_layout.addWidget(generate_btn)
        main_layout.addLayout(button_layout)

        # 初始状态
        self._on_rotation_toggled(self.enable_rotation_check.isChecked())
        self._on_scale_toggled(self.enable_scale_check.isChecked())
        self._on_blend_toggled(self.enable_blend_check.isChecked())
        self._on_color_toggled(self.enable_color_check.isChecked())
        self._sync_range_to_visual()

    # --- 状态同步 ---
    def _on_visual_range_changed(self, x_min: int, y_min: int, x_max: int, y_max: int):
        self.x_min_spin.blockSignals(True)
        self.y_min_spin.blockSignals(True)
        self.x_max_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)
        self.x_min_spin.setValue(x_min)
        self.y_min_spin.setValue(y_min)
        self.x_max_spin.setValue(x_max)
        self.y_max_spin.setValue(y_max)
        self.x_min_spin.blockSignals(False)
        self.y_min_spin.blockSignals(False)
        self.x_max_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)

    def _on_numeric_range_changed(self):
        self._sync_range_to_visual()

    def _sync_range_to_visual(self):
        self.range_selector.set_range(
            self.x_min_spin.value(),
            self.y_min_spin.value(),
            self.x_max_spin.value(),
            self.y_max_spin.value(),
        )

    def _reset_range(self):
        self.x_min_spin.setValue(int(self.canvas_width * 0.1))
        self.y_min_spin.setValue(int(self.canvas_height * 0.1))
        self.x_max_spin.setValue(int(self.canvas_width * 0.9))
        self.y_max_spin.setValue(int(self.canvas_height * 0.9))
        self._sync_range_to_visual()

    def _on_rotation_toggled(self, enabled: bool):
        self.rotation_min_spin.setEnabled(enabled)
        self.rotation_max_spin.setEnabled(enabled)

    def _on_scale_toggled(self, enabled: bool):
        self.scale_min_spin.setEnabled(enabled)
        self.scale_max_spin.setEnabled(enabled)

    def _on_blend_toggled(self, enabled: bool):
        self.blend_normal_check.setEnabled(enabled)
        self.blend_poisson_normal_check.setEnabled(enabled)
        self.blend_poisson_mixed_check.setEnabled(enabled)

    def _on_color_toggled(self, enabled: bool):
        self.color_random_check.setEnabled(enabled)
        self.color_background_check.setEnabled(enabled)
        self.color_preset_check.setEnabled(enabled)
        self.color_preset_combo.setEnabled(enabled)
        self.color_opacity_min_spinbox.setEnabled(enabled)
        self.color_opacity_max_spinbox.setEnabled(enabled)

    # --- 生成参数 ---
    def get_generation_params(self) -> dict:
        return {
            "count": self.count_spin.value(),
            "mode": self.mode_combo.currentIndex(),
            "position": {
                "x_range": (self.x_min_spin.value(), self.x_max_spin.value()),
                "y_range": (self.y_min_spin.value(), self.y_max_spin.value()),
                "avoid_overlap": self.avoid_overlap_check.isChecked(),
            },
            "blend": {
                "enabled": self.enable_blend_check.isChecked(),
                "modes": self._get_enabled_blend_modes(),
            },
            "rotation": {
                "enabled": self.enable_rotation_check.isChecked(),
                "range": (
                    self.rotation_min_spin.value(),
                    self.rotation_max_spin.value(),
                ),
            },
            "scale": {
                "enabled": self.enable_scale_check.isChecked(),
                "range": (
                    self.scale_min_spin.value() / 100.0,
                    self.scale_max_spin.value() / 100.0,
                ),
            },
            "color_overlay": {
                "enabled": self.enable_color_check.isChecked(),
                "use_background_colors": self.color_background_check.isChecked(),
                "use_preset": self.color_preset_check.isChecked(),
                "preset_group": self.color_preset_combo.currentText(),
                "opacity_range": (
                    self.color_opacity_min_spinbox.value() / 100.0,
                    self.color_opacity_max_spinbox.value() / 100.0,
                ),
            },
        }

    def _get_enabled_blend_modes(self) -> List[str]:
        modes: List[str] = []
        if self.enable_blend_check.isChecked():
            if self.blend_normal_check.isChecked():
                modes.append("normal")
            if self.blend_poisson_normal_check.isChecked():
                modes.append("poisson_normal")
            if self.blend_poisson_mixed_check.isChecked():
                modes.append("poisson_mixed")
        if not modes:
            modes = ["normal"]
        return modes

    # --- 颜色预设 & 随机颜色 ---
    def _get_color_presets(self) -> dict:
        return {
            "暖色调": [
                (255, 69, 0),
                (255, 140, 0),
                (255, 165, 0),
                (255, 215, 0),
                (255, 20, 147),
                (220, 20, 60),
            ],
            "冷色调": [
                (0, 191, 255),
                (30, 144, 255),
                (0, 206, 209),
                (64, 224, 208),
                (72, 61, 139),
                (123, 104, 238),
            ],
            "大地色": [
                (139, 69, 19),
                (160, 82, 45),
                (210, 180, 140),
                (222, 184, 135),
                (205, 133, 63),
                (244, 164, 96),
            ],
            "彩虹色": [
                (255, 0, 0),
                (255, 165, 0),
                (255, 255, 0),
                (0, 255, 0),
                (0, 0, 255),
                (75, 0, 130),
                (238, 130, 238),
            ],
            "单色调-红": [
                (255, 182, 193),
                (255, 105, 180),
                (255, 20, 147),
                (220, 20, 60),
                (178, 34, 34),
                (139, 0, 0),
            ],
            "单色调-蓝": [
                (173, 216, 230),
                (135, 206, 235),
                (0, 191, 255),
                (30, 144, 255),
                (0, 0, 255),
                (0, 0, 139),
            ],
            "单色调-绿": [
                (144, 238, 144),
                (152, 251, 152),
                (0, 255, 127),
                (0, 250, 154),
                (0, 255, 0),
                (0, 128, 0),
            ],
        }

    def generate_random_color(
        self, preset_name: str | None = None
    ) -> Tuple[int, int, int]:
        if preset_name and self.color_preset_check.isChecked():
            presets = self._get_color_presets()
            if preset_name in presets:
                return random.choice(presets[preset_name])
        # 完全随机
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
