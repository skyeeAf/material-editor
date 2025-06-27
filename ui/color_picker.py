"""
背景取色器工具
用于从背景图像中提取颜色
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ColorSwatch(QFrame):
    """颜色色块组件"""

    color_selected = Signal(tuple)  # RGB颜色被选中

    def __init__(self, color: Tuple[int, int, int], parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(30, 30)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # 设置背景颜色
        self.setStyleSheet(f"""
            QFrame {{
                background-color: rgb({color[0]}, {color[1]}, {color[2]});
                border: 2px solid #cccccc;
                border-radius: 4px;
            }}
            QFrame:hover {{
                border: 2px solid #0078d4;
            }}
        """)

        # 工具提示显示RGB值
        self.setToolTip(f"RGB({color[0]}, {color[1]}, {color[2]})")

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.color_selected.emit(self.color)


class BackgroundColorPicker(QWidget):
    """背景取色器工具"""

    color_picked = Signal(tuple)  # 颜色被选择时发出信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.background_image: Optional[np.ndarray] = None
        self.extracted_colors: List[Tuple[int, int, int]] = []
        self.sampling_enabled = False
        self._init_ui()

    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 标题
        title_label = QLabel("背景取色器")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)

        # 控制按钮
        button_layout = QHBoxLayout()

        self.sample_btn = QPushButton("开始取色")
        self.sample_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.sample_btn.clicked.connect(self._toggle_sampling)
        button_layout.addWidget(self.sample_btn)

        self.clear_btn = QPushButton("清除")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.clear_btn.clicked.connect(self._clear_colors)
        button_layout.addWidget(self.clear_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # 说明文字
        info_label = QLabel("点击'开始取色'后，在画布上点击提取颜色")
        info_label.setStyleSheet("color: #666666; font-size: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 颜色显示区域
        colors_label = QLabel("提取的颜色:")
        colors_label.setStyleSheet(
            "font-weight: bold; font-size: 11px; margin-top: 8px;"
        )
        layout.addWidget(colors_label)

        # 滚动区域包含颜色网格
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(120)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.colors_widget = QWidget()
        self.colors_layout = QGridLayout(self.colors_widget)
        self.colors_layout.setSpacing(4)
        scroll_area.setWidget(self.colors_widget)
        layout.addWidget(scroll_area)

        # 快速操作
        quick_layout = QHBoxLayout()

        auto_extract_btn = QPushButton("自动提取")
        auto_extract_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        auto_extract_btn.clicked.connect(self._auto_extract_colors)
        quick_layout.addWidget(auto_extract_btn)

        custom_color_btn = QPushButton("自定义")
        custom_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        custom_color_btn.clicked.connect(self._add_custom_color)
        quick_layout.addWidget(custom_color_btn)

        quick_layout.addStretch()
        layout.addLayout(quick_layout)

        layout.addStretch()

    def set_background_image(self, image: np.ndarray):
        """设置背景图像"""
        self.background_image = image

    def _toggle_sampling(self):
        """切换取色模式"""
        self.sampling_enabled = not self.sampling_enabled
        if self.sampling_enabled:
            self.sample_btn.setText("停止取色")
            self.sample_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """)
        else:
            self.sample_btn.setText("开始取色")
            self.sample_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
            """)

    def is_sampling_enabled(self) -> bool:
        """检查是否正在取色"""
        return self.sampling_enabled

    def sample_color_at_position(self, x: int, y: int):
        """在指定位置取色"""
        if self.background_image is None or not self.sampling_enabled:
            print(
                f"取色失败: background_image={self.background_image is not None}, sampling_enabled={self.sampling_enabled}"
            )
            return

        try:
            # 确保坐标在图像范围内
            h, w = self.background_image.shape[:2]
            print(f"取色位置: ({x}, {y}), 图像尺寸: {w}x{h}")

            if 0 <= x < w and 0 <= y < h:
                # 获取BGR颜色并转换为RGB
                bgr_color = self.background_image[y, x]
                rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
                print(f"提取到颜色: RGB{rgb_color}")

                # 避免重复颜色
                if rgb_color not in self.extracted_colors:
                    self.extracted_colors.append(rgb_color)
                    self._add_color_swatch(rgb_color)
                    print(f"添加新颜色: RGB{rgb_color}")
                else:
                    print(f"颜色已存在: RGB{rgb_color}")
            else:
                print(
                    f"坐标超出图像范围: ({x}, {y}) 不在 (0, 0) 到 ({w - 1}, {h - 1}) 范围内"
                )

        except Exception as e:
            print(f"取色失败: {e}")
            import traceback

            traceback.print_exc()

    def _add_color_swatch(self, color: Tuple[int, int, int]):
        """添加颜色色块"""
        swatch = ColorSwatch(color)
        swatch.color_selected.connect(self.color_picked.emit)

        # 计算网格位置
        count = len(self.extracted_colors) - 1
        row = count // 6  # 每行6个
        col = count % 6

        self.colors_layout.addWidget(swatch, row, col)

    def _clear_colors(self):
        """清除所有提取的颜色"""
        self.extracted_colors.clear()

        # 清除所有色块组件
        while self.colors_layout.count():
            child = self.colors_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _auto_extract_colors(self):
        """自动提取背景中的主要颜色"""
        if self.background_image is None:
            QMessageBox.information(self, "提示", "请先加载背景图像")
            return

        try:
            print(f"开始自动提取颜色，背景图像尺寸: {self.background_image.shape}")

            # 使用K-means聚类提取主要颜色
            image = self.background_image.copy()

            # 缩小图像以提高处理速度
            h, w = image.shape[:2]
            if h > 300 or w > 300:
                scale = min(300 / h, 300 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))

            # 重塑为像素数组
            pixels = image.reshape(-1, 3).astype(np.float32)

            # K-means聚类
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = min(8, len(pixels))  # 最多提取8种颜色

            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            # 转换颜色并添加
            added_count = 0
            for center in centers:
                bgr_color = center.astype(int)
                rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))

                if rgb_color not in self.extracted_colors:
                    self.extracted_colors.append(rgb_color)
                    self._add_color_swatch(rgb_color)
                    added_count += 1

            print(f"自动提取完成，添加了 {added_count} 种新颜色")

        except Exception as e:
            print(f"自动提取颜色失败: {e}")
            import traceback

            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"自动提取颜色失败: {e}")

    def _add_custom_color(self):
        """添加自定义颜色"""
        color_dialog = QColorDialog(self)
        color_dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)

        if color_dialog.exec() == QColorDialog.DialogCode.Accepted:
            qcolor = color_dialog.selectedColor()
            rgb_color = (qcolor.red(), qcolor.green(), qcolor.blue())

            if rgb_color not in self.extracted_colors:
                self.extracted_colors.append(rgb_color)
                self._add_color_swatch(rgb_color)

    def get_extracted_colors(self) -> List[Tuple[int, int, int]]:
        """获取所有提取的颜色"""
        return self.extracted_colors.copy()

    def get_random_extracted_color(self) -> Optional[Tuple[int, int, int]]:
        """获取一个随机的提取颜色"""
        if not self.extracted_colors:
            return None
        import random

        return random.choice(self.extracted_colors)
