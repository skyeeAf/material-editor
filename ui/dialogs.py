"""
对话框组件
包含各种设置和配置对话框
"""

import math
import random
from typing import List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class VisualRangeSelector(QWidget):
    """可视化范围选择器"""

    range_changed = Signal(int, int, int, int)  # x_min, y_min, x_max, y_max

    def __init__(
        self, canvas_width: int, canvas_height: int, background_image=None, parent=None
    ):
        super().__init__(parent)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        # 减小预览区域尺寸，使对话框更紧凑
        self.preview_width = 250
        self.preview_height = 140

        # 背景图像
        self.background_image = background_image
        self.background_pixmap = None

        # 如果有背景图像，创建缩放后的预览图
        if self.background_image is not None:
            self._create_background_preview()

        # 当前选择的范围（相对于画布的比例，0-1）
        self.x_min_ratio = 0.1
        self.y_min_ratio = 0.1
        self.x_max_ratio = 0.9
        self.y_max_ratio = 0.9

        # 拖拽状态
        self.dragging = False
        self.drag_corner = (
            None  # 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'move'
        )
        self.last_mouse_pos = None

        self.setFixedSize(self.preview_width, self.preview_height)
        self.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")

    def _create_background_preview(self):
        """创建背景图像的预览"""
        try:
            import cv2

            from utils.image_utils import numpy_to_qimage

            # 直接缩放到预览区域大小，不保持宽高比
            # 这样用户可以在整个预览区域选择位置
            resized_img = cv2.resize(
                self.background_image,
                (self.preview_width, self.preview_height),
                interpolation=cv2.INTER_AREA,
            )

            # 转换为QPixmap
            qimage = numpy_to_qimage(resized_img)
            if qimage:
                self.background_pixmap = QPixmap.fromImage(qimage)

        except Exception as e:
            print(f"创建背景预览失败: {e}")
            self.background_pixmap = None

    def set_range(self, x_min: int, y_min: int, x_max: int, y_max: int):
        """设置范围"""
        self.x_min_ratio = x_min / self.canvas_width
        self.y_min_ratio = y_min / self.canvas_height
        self.x_max_ratio = x_max / self.canvas_width
        self.y_max_ratio = y_max / self.canvas_height
        self.update()

    def get_range(self) -> Tuple[int, int, int, int]:
        """获取当前范围"""
        x_min = int(self.x_min_ratio * self.canvas_width)
        y_min = int(self.y_min_ratio * self.canvas_height)
        x_max = int(self.x_max_ratio * self.canvas_width)
        y_max = int(self.y_max_ratio * self.canvas_height)
        return x_min, y_min, x_max, y_max

    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制背景图像（如果有）
        if self.background_pixmap:
            # 直接填充整个预览区域
            painter.drawPixmap(
                0, 0, self.preview_width, self.preview_height, self.background_pixmap
            )
        else:
            # 绘制背景网格（如果没有背景图像）
            painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.PenStyle.DotLine))
            for i in range(0, self.preview_width, 20):
                painter.drawLine(i, 0, i, self.preview_height)
            for i in range(0, self.preview_height, 20):
                painter.drawLine(0, i, self.preview_width, i)

        # 计算选择区域在预览中的位置
        x1 = int(self.x_min_ratio * self.preview_width)
        y1 = int(self.y_min_ratio * self.preview_height)
        x2 = int(self.x_max_ratio * self.preview_width)
        y2 = int(self.y_max_ratio * self.preview_height)

        # 绘制未选择区域（半透明灰色）
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(QColor(128, 128, 128, 150)))

        # 上方区域
        if y1 > 0:
            painter.drawRect(0, 0, self.preview_width, y1)
        # 下方区域
        if y2 < self.preview_height:
            painter.drawRect(0, y2, self.preview_width, self.preview_height - y2)
        # 左侧区域
        if x1 > 0:
            painter.drawRect(0, y1, x1, y2 - y1)
        # 右侧区域
        if x2 < self.preview_width:
            painter.drawRect(x2, y1, self.preview_width - x2, y2 - y1)

        # 绘制选择区域边框
        painter.setPen(QPen(QColor(0, 120, 215), 2))
        painter.setBrush(QBrush())
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        # 绘制控制点
        handle_size = 6
        painter.setPen(QPen(QColor(0, 120, 215), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255)))

        # 四个角的控制点
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

        # 绘制尺寸标签（使用对比色）
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))

        # 左上角坐标标签
        top_text = f"{int(self.x_min_ratio * self.canvas_width)}, {int(self.y_min_ratio * self.canvas_height)}"
        text_rect1 = painter.fontMetrics().boundingRect(top_text)
        text_rect1.moveTopLeft(QPoint(x1 + 2, y1 + 2))
        painter.drawRect(text_rect1.adjusted(-2, -1, 2, 1))
        painter.drawText(text_rect1, Qt.AlignmentFlag.AlignLeft, top_text)

        # 右下角坐标标签
        bottom_text = f"{int(self.x_max_ratio * self.canvas_width)}, {int(self.y_max_ratio * self.canvas_height)}"
        text_rect2 = painter.fontMetrics().boundingRect(bottom_text)
        text_rect2.moveBottomRight(QPoint(x2 - 2, y2 - 2))
        painter.drawRect(text_rect2.adjusted(-2, -1, 2, 1))
        painter.drawText(text_rect2, Qt.AlignmentFlag.AlignRight, bottom_text)

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            x, y = pos.x(), pos.y()

            x1 = int(self.x_min_ratio * self.preview_width)
            y1 = int(self.y_min_ratio * self.preview_height)
            x2 = int(self.x_max_ratio * self.preview_width)
            y2 = int(self.y_max_ratio * self.preview_height)

            handle_size = 8

            # 检查是否点击了控制点
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
        """鼠标移动事件"""
        if self.dragging and self.last_mouse_pos:
            pos = event.position().toPoint()
            dx = pos.x() - self.last_mouse_pos.x()
            dy = pos.y() - self.last_mouse_pos.y()

            # 转换为比例变化
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
                # 移动整个选择区域
                width_ratio = self.x_max_ratio - self.x_min_ratio
                height_ratio = self.y_max_ratio - self.y_min_ratio

                new_x_min = self.x_min_ratio + dx_ratio
                new_y_min = self.y_min_ratio + dy_ratio

                # 确保不超出边界
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

            # 发送范围改变信号
            x_min, y_min, x_max, y_max = self.get_range()
            self.range_changed.emit(x_min, y_min, x_max, y_max)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.drag_corner = None
            self.last_mouse_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)


class RandomGenerateDialog(QDialog):
    """随机生成物体对话框"""

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
        self.setWindowTitle("随机生成物体")
        self.setModal(True)
        # 调整对话框尺寸以适应平铺布局
        self.resize(650, 800)
        self.setMaximumHeight(900)
        self.setMinimumHeight(700)
        self.setMinimumWidth(600)
        self._init_ui()

    def _init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        main_layout.addWidget(scroll_area)

        # 创建内容容器
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)

        # 基本设置组
        basic_group = QGroupBox("基本设置")
        basic_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
        """)
        basic_layout = QFormLayout(basic_group)
        basic_layout.setSpacing(8)
        basic_layout.setHorizontalSpacing(15)

        # 预设配置
        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            [
                "自定义",
                "少量随机(5个)",
                "中等随机(15个)",
                "大量随机(30个)",
                "均匀分布",
                "边缘分布",
                "中心聚集",
            ]
        )
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)

        preset_info_btn = QPushButton("?")
        preset_info_btn.setMaximumSize(30, 30)
        preset_info_btn.setStyleSheet("""
            QPushButton {
                border-radius: 15px;
                background-color: #e1e1e1;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
        preset_info_btn.clicked.connect(self._show_preset_info)
        preset_layout.addWidget(preset_info_btn)
        preset_layout.addStretch()
        basic_layout.addRow("预设配置:", preset_layout)

        # 生成数量和模式
        count_mode_layout = QHBoxLayout()
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100)
        self.count_spin.setValue(10)
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

        # 位置设置组
        position_group = QGroupBox("位置设置")
        position_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
        """)
        position_layout = QVBoxLayout(position_group)
        position_layout.setSpacing(10)

        # 位置设置的上半部分：可视化选择器和重置按钮
        visual_section = QVBoxLayout()

        # 可视化选择器标题和重置按钮
        range_header = QHBoxLayout()
        range_header.addWidget(QLabel("生成范围预览:"))
        range_header.addStretch()
        reset_range_btn = QPushButton("重置范围")
        reset_range_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        reset_range_btn.clicked.connect(self._reset_range)
        range_header.addWidget(reset_range_btn)
        visual_section.addLayout(range_header)

        # 创建可视化选择器
        self.range_selector = VisualRangeSelector(
            self.canvas_width, self.canvas_height, self.background_image
        )
        self.range_selector.range_changed.connect(self._on_visual_range_changed)
        visual_section.addWidget(self.range_selector, 0, Qt.AlignmentFlag.AlignCenter)

        position_layout.addLayout(visual_section)

        # 位置设置的下半部分：数值输入和选项
        controls_section = QVBoxLayout()
        controls_section.setSpacing(8)

        # X和Y范围输入框 - 分成两行，避免与可视化区域冲突
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

        # Y范围输入框
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

        # 避免重叠选项
        self.avoid_overlap_check = QCheckBox("尝试避免重叠")
        self.avoid_overlap_check.setChecked(True)
        controls_section.addWidget(self.avoid_overlap_check)

        position_layout.addLayout(controls_section)
        layout.addWidget(position_group)

        # 变换设置组
        transform_group = QGroupBox("变换设置")
        transform_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
        """)
        transform_layout = QFormLayout(transform_group)
        transform_layout.setSpacing(8)
        transform_layout.setHorizontalSpacing(15)

        # 旋转设置
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
        self.rotation_min_spin.setMinimumWidth(80)
        rotation_layout.addWidget(self.rotation_min_spin)

        rotation_layout.addWidget(QLabel("到"))
        self.rotation_max_spin = QSpinBox()
        self.rotation_max_spin.setRange(-360, 360)
        self.rotation_max_spin.setValue(360)
        self.rotation_max_spin.setMinimumWidth(80)
        rotation_layout.addWidget(self.rotation_max_spin)
        rotation_layout.addWidget(QLabel("度"))
        rotation_layout.addStretch()
        transform_layout.addRow("旋转:", rotation_layout)

        # 缩放设置
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
        self.scale_min_spin.setMinimumWidth(80)
        scale_layout.addWidget(self.scale_min_spin)

        scale_layout.addWidget(QLabel("到"))
        self.scale_max_spin = QSpinBox()
        self.scale_max_spin.setRange(10, 500)
        self.scale_max_spin.setValue(150)
        self.scale_max_spin.setSuffix("%")
        self.scale_max_spin.setMinimumWidth(80)
        scale_layout.addWidget(self.scale_max_spin)
        scale_layout.addStretch()
        transform_layout.addRow("缩放:", scale_layout)

        layout.addWidget(transform_group)

        # 混合模式设置组
        blend_group = QGroupBox("混合模式设置")
        blend_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
        """)
        blend_layout = QFormLayout(blend_group)
        blend_layout.setSpacing(8)
        blend_layout.setHorizontalSpacing(15)

        # 启用混合模式
        self.enable_blend_check = QCheckBox("启用随机混合模式")
        self.enable_blend_check.setChecked(False)
        self.enable_blend_check.toggled.connect(self._on_blend_toggled)
        blend_layout.addRow("", self.enable_blend_check)

        # 混合模式选择
        blend_modes_layout = QHBoxLayout()
        self.blend_normal_check = QCheckBox("普通")
        self.blend_normal_check.setChecked(True)
        blend_modes_layout.addWidget(self.blend_normal_check)

        self.blend_poisson_normal_check = QCheckBox("泊松融合(正常) - 推荐")
        self.blend_poisson_normal_check.setChecked(True)
        self.blend_poisson_normal_check.setStyleSheet(
            "font-weight: bold; color: #0078d4;"
        )
        blend_modes_layout.addWidget(self.blend_poisson_normal_check)

        self.blend_poisson_mixed_check = QCheckBox("泊松融合(混合)")
        self.blend_poisson_mixed_check.setChecked(False)
        blend_modes_layout.addWidget(self.blend_poisson_mixed_check)

        blend_modes_layout.addStretch()
        blend_layout.addRow("可选模式:", blend_modes_layout)

        layout.addWidget(blend_group)

        # 色彩叠加设置组
        color_group = QGroupBox("色彩叠加设置")
        color_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
        """)
        color_layout = QFormLayout(color_group)
        color_layout.setSpacing(8)
        color_layout.setHorizontalSpacing(15)

        # 启用色彩叠加
        self.enable_color_check = QCheckBox("启用随机色彩叠加")
        self.enable_color_check.setChecked(False)
        self.enable_color_check.toggled.connect(self._on_color_toggled)
        color_layout.addRow("", self.enable_color_check)

        # 颜色模式选择
        color_mode_layout = QVBoxLayout()
        color_mode_layout.setSpacing(5)

        self.color_random_check = QCheckBox("随机颜色")
        self.color_random_check.setChecked(True)
        color_mode_layout.addWidget(self.color_random_check)

        # 随机背景色选项
        self.color_background_check = QCheckBox("随机背景色（基于生成位置）")
        self.color_background_check.setChecked(False)
        self.color_background_check.setToolTip("从生成位置附近的背景颜色中随机选择")
        color_mode_layout.addWidget(self.color_background_check)

        # 预设颜色组
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

        # 透明度范围
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("透明度范围:"))
        self.color_opacity_min_spinbox = QSpinBox()
        self.color_opacity_min_spinbox.setRange(10, 100)
        self.color_opacity_min_spinbox.setValue(20)
        self.color_opacity_min_spinbox.setSuffix("%")
        self.color_opacity_min_spinbox.setMinimumWidth(80)
        opacity_layout.addWidget(self.color_opacity_min_spinbox)

        opacity_layout.addWidget(QLabel("到"))
        self.color_opacity_max_spinbox = QSpinBox()
        self.color_opacity_max_spinbox.setRange(10, 100)
        self.color_opacity_max_spinbox.setValue(60)
        self.color_opacity_max_spinbox.setSuffix("%")
        self.color_opacity_max_spinbox.setMinimumWidth(80)
        opacity_layout.addWidget(self.color_opacity_max_spinbox)
        opacity_layout.addStretch()
        color_layout.addRow("", opacity_layout)

        layout.addWidget(color_group)

        # 按钮区域 - 固定在主布局底部
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setMinimumSize(80, 35)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        generate_btn = QPushButton("生成")
        generate_btn.setMinimumSize(80, 35)
        generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: 1px solid #0078d4;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        generate_btn.clicked.connect(self.accept)
        button_layout.addWidget(generate_btn)

        main_layout.addLayout(button_layout)

        # 初始状态
        self._on_rotation_toggled(self.enable_rotation_check.isChecked())
        self._on_scale_toggled(self.enable_scale_check.isChecked())
        self._on_blend_toggled(self.enable_blend_check.isChecked())
        self._on_color_toggled(self.enable_color_check.isChecked())

        # 同步初始范围
        self._sync_range_to_visual()

    def _on_visual_range_changed(self, x_min: int, y_min: int, x_max: int, y_max: int):
        """可视化范围改变事件"""
        # 更新数值输入框
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
        """数值范围改变事件"""
        # 更新可视化选择器
        self.range_selector.set_range(
            self.x_min_spin.value(),
            self.y_min_spin.value(),
            self.x_max_spin.value(),
            self.y_max_spin.value(),
        )

    def _sync_range_to_visual(self):
        """同步范围到可视化选择器"""
        self.range_selector.set_range(
            self.x_min_spin.value(),
            self.y_min_spin.value(),
            self.x_max_spin.value(),
            self.y_max_spin.value(),
        )

    def _reset_range(self):
        """重置范围到默认值"""
        self.x_min_spin.setValue(int(self.canvas_width * 0.1))
        self.y_min_spin.setValue(int(self.canvas_height * 0.1))
        self.x_max_spin.setValue(int(self.canvas_width * 0.9))
        self.y_max_spin.setValue(int(self.canvas_height * 0.9))
        self._sync_range_to_visual()

    def _on_rotation_toggled(self, enabled: bool):
        """旋转设置切换"""
        self.rotation_min_spin.setEnabled(enabled)
        self.rotation_max_spin.setEnabled(enabled)

    def _on_scale_toggled(self, enabled: bool):
        """缩放设置切换"""
        self.scale_min_spin.setEnabled(enabled)
        self.scale_max_spin.setEnabled(enabled)

    def _on_blend_toggled(self, enabled: bool):
        """混合模式设置切换"""
        self.blend_normal_check.setEnabled(enabled)
        self.blend_poisson_normal_check.setEnabled(enabled)
        self.blend_poisson_mixed_check.setEnabled(enabled)

    def _on_color_toggled(self, enabled: bool):
        """色彩叠加设置切换"""
        self.color_random_check.setEnabled(enabled)
        self.color_preset_check.setEnabled(enabled)
        self.color_preset_combo.setEnabled(enabled)
        self.color_opacity_min_spinbox.setEnabled(enabled)
        self.color_opacity_max_spinbox.setEnabled(enabled)

    def _on_preset_changed(self, index: int):
        """预设配置改变"""
        if index == 0:  # 自定义
            return
        elif index == 1:  # 少量随机(5个)
            self._apply_preset_config(
                {
                    "count": 5,
                    "mode": 0,
                    "position": {
                        "x_ratio": (0.1, 0.9),
                        "y_ratio": (0.1, 0.9),
                        "avoid_overlap": True,
                    },
                    "rotation": {"enabled": True, "range": (0, 360)},
                    "scale": {"enabled": True, "range": (0.8, 1.2)},
                    "blend": {"enabled": False},
                }
            )
        elif index == 2:  # 中等随机(15个)
            self._apply_preset_config(
                {
                    "count": 15,
                    "mode": 0,
                    "position": {
                        "x_ratio": (0.05, 0.95),
                        "y_ratio": (0.05, 0.95),
                        "avoid_overlap": True,
                    },
                    "rotation": {"enabled": True, "range": (0, 360)},
                    "scale": {"enabled": True, "range": (0.6, 1.4)},
                    "blend": {"enabled": True},
                }
            )
        elif index == 3:  # 大量随机(30个)
            self._apply_preset_config(
                {
                    "count": 30,
                    "mode": 0,
                    "position": {
                        "x_ratio": (0.02, 0.98),
                        "y_ratio": (0.02, 0.98),
                        "avoid_overlap": False,
                    },
                    "rotation": {"enabled": True, "range": (0, 360)},
                    "scale": {"enabled": True, "range": (0.5, 1.5)},
                    "blend": {"enabled": True},
                }
            )
        elif index == 4:  # 均匀分布
            self._apply_preset_config(
                {
                    "count": 12,
                    "mode": 3,
                    "position": {
                        "x_ratio": (0.1, 0.9),
                        "y_ratio": (0.1, 0.9),
                        "avoid_overlap": True,
                    },
                    "rotation": {"enabled": False, "range": (0, 0)},
                    "scale": {"enabled": False, "range": (1.0, 1.0)},
                    "blend": {"enabled": False},
                }
            )
        elif index == 5:  # 边缘分布
            self._apply_preset_config(
                {
                    "count": 20,
                    "mode": 0,
                    "position": {
                        "x_ratio": (0.0, 1.0),
                        "y_ratio": (0.0, 1.0),
                        "avoid_overlap": True,
                        "edge_bias": True,
                    },
                    "rotation": {"enabled": True, "range": (-45, 45)},
                    "scale": {"enabled": True, "range": (0.7, 1.3)},
                    "blend": {"enabled": True},
                }
            )
        elif index == 6:  # 中心聚集
            self._apply_preset_config(
                {
                    "count": 10,
                    "mode": 0,
                    "position": {
                        "x_ratio": (0.3, 0.7),
                        "y_ratio": (0.3, 0.7),
                        "avoid_overlap": True,
                    },
                    "rotation": {"enabled": True, "range": (0, 360)},
                    "scale": {"enabled": True, "range": (0.8, 1.2)},
                    "blend": {"enabled": False},
                }
            )

    def _apply_preset_config(self, config: dict):
        """应用预设配置"""
        # 设置数量
        self.count_spin.setValue(config["count"])

        # 设置模式
        self.mode_combo.setCurrentIndex(config["mode"])

        # 设置位置
        pos_config = config["position"]
        self.x_min_spin.setValue(int(self.canvas_width * pos_config["x_ratio"][0]))
        self.x_max_spin.setValue(int(self.canvas_width * pos_config["x_ratio"][1]))
        self.y_min_spin.setValue(int(self.canvas_height * pos_config["y_ratio"][0]))
        self.y_max_spin.setValue(int(self.canvas_height * pos_config["y_ratio"][1]))
        self.avoid_overlap_check.setChecked(pos_config["avoid_overlap"])

        # 同步到可视化选择器
        self._sync_range_to_visual()

        # 设置旋转
        rot_config = config["rotation"]
        self.enable_rotation_check.setChecked(rot_config["enabled"])
        self.rotation_min_spin.setValue(rot_config["range"][0])
        self.rotation_max_spin.setValue(rot_config["range"][1])

        # 设置缩放
        scale_config = config["scale"]
        self.enable_scale_check.setChecked(scale_config["enabled"])
        self.scale_min_spin.setValue(int(scale_config["range"][0] * 100))
        self.scale_max_spin.setValue(int(scale_config["range"][1] * 100))

        # 设置混合模式
        blend_config = config["blend"]
        self.enable_blend_check.setChecked(blend_config["enabled"])

    def _show_preset_info(self):
        """显示预设配置信息"""
        from PySide6.QtWidgets import QMessageBox

        info_text = """预设配置说明：

• 少量随机(5个): 生成5个随机素材，启用旋转和轻微缩放
• 中等随机(15个): 生成15个随机素材，更大的变化范围，启用混合模式
• 大量随机(30个): 生成30个随机素材，允许重叠，最大变化，启用混合模式
• 均匀分布: 所有素材类型均匀分布，无旋转缩放和混合模式
• 边缘分布: 素材主要分布在画布边缘区域，启用混合模式
• 中心聚集: 素材集中在画布中心区域

选择预设后可以进一步调整参数。
可视化预览区域支持鼠标拖拽调整范围。"""

        QMessageBox.information(self, "预设配置说明", info_text)

    def get_generation_params(self) -> dict:
        """获取生成参数"""
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
        """获取启用的混合模式列表"""
        modes = []
        if self.enable_blend_check.isChecked():
            if self.blend_normal_check.isChecked():
                modes.append("normal")
            if self.blend_poisson_normal_check.isChecked():
                modes.append("poisson_normal")
            if self.blend_poisson_mixed_check.isChecked():
                modes.append("poisson_mixed")

        # 如果没有选择任何混合模式，默认使用普通模式
        if not modes:
            modes = ["normal"]

        return modes

    def _get_color_presets(self) -> dict:
        """获取预设颜色组"""
        return {
            "暖色调": [
                (255, 69, 0),  # 橙红
                (255, 140, 0),  # 深橙
                (255, 165, 0),  # 橙色
                (255, 215, 0),  # 金色
                (255, 20, 147),  # 深粉红
                (220, 20, 60),  # 深红
            ],
            "冷色调": [
                (0, 191, 255),  # 深天蓝
                (30, 144, 255),  # 道奇蓝
                (0, 206, 209),  # 深绿松石
                (64, 224, 208),  # 绿松石
                (72, 61, 139),  # 深石板蓝
                (123, 104, 238),  # 中石板蓝
            ],
            "大地色": [
                (139, 69, 19),  # 马鞍棕
                (160, 82, 45),  # 马鞍棕
                (210, 180, 140),  # 棕褐
                (222, 184, 135),  # 浅黄褐
                (205, 133, 63),  # 秘鲁
                (244, 164, 96),  # 沙棕
            ],
            "彩虹色": [
                (255, 0, 0),  # 红
                (255, 165, 0),  # 橙
                (255, 255, 0),  # 黄
                (0, 255, 0),  # 绿
                (0, 0, 255),  # 蓝
                (75, 0, 130),  # 靛
                (238, 130, 238),  # 紫
            ],
            "单色调-红": [
                (255, 182, 193),  # 浅粉红
                (255, 105, 180),  # 热粉红
                (255, 20, 147),  # 深粉红
                (220, 20, 60),  # 深红
                (178, 34, 34),  # 火砖红
                (139, 0, 0),  # 深红
            ],
            "单色调-蓝": [
                (173, 216, 230),  # 浅蓝
                (135, 206, 235),  # 天蓝
                (0, 191, 255),  # 深天蓝
                (30, 144, 255),  # 道奇蓝
                (0, 0, 255),  # 蓝
                (0, 0, 139),  # 深蓝
            ],
            "单色调-绿": [
                (144, 238, 144),  # 浅绿
                (152, 251, 152),  # 苍绿
                (0, 255, 127),  # 春绿
                (0, 250, 154),  # 中春绿
                (0, 255, 0),  # 绿
                (0, 128, 0),  # 绿
            ],
        }

    def generate_random_color(self, preset_name: str = None) -> Tuple[int, int, int]:
        """生成随机颜色"""
        if preset_name and self.color_preset_check.isChecked():
            presets = self._get_color_presets()
            if preset_name in presets:
                return random.choice(presets[preset_name])

        # 生成完全随机的颜色
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
