"""
画布组件
支持素材的拖拽、缩放、旋转等操作
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QLabel, QScrollArea, QWidget

from core.layer import LayerManager
from core.material import MaterialInstance
from utils.image_utils import numpy_to_qimage


class CanvasWidget(QLabel):
    """画布组件"""

    # 信号定义
    instance_selected = Signal(object)  # 素材实例被选中
    instance_moved = Signal(object)  # 素材实例被移动
    instance_transformed = Signal(object)  # 素材实例被变换
    canvas_clicked = Signal(int, int)  # 画布被单击
    canvas_double_clicked = Signal(int, int)  # 画布被双击

    def __init__(self, parent=None):
        super().__init__(parent)

        # 画布属性
        self.background_image: Optional[np.ndarray] = None
        self.layer_manager: Optional[LayerManager] = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # 画布视口偏移（用于拖动）
        self.canvas_offset = QPoint(0, 0)

        # 当前合成的图像
        self.composite_image: Optional[np.ndarray] = None
        self.composite_pixmap: Optional[QPixmap] = None

        # 交互状态
        self.selected_instance: Optional[MaterialInstance] = None
        self.dragging = False
        self.canvas_dragging = False  # 画布拖动状态
        self.drag_start_pos = QPoint()
        self.drag_offset = QPoint()
        self.canvas_drag_start_offset = QPoint()  # 画布拖动开始时的偏移

        # 双击检测
        self.click_timer = QTimer()
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self._handle_single_click)
        self.pending_click_pos = None
        self.double_click_threshold = 300  # 毫秒

        # 变换控制
        self.show_bounding_boxes = True
        self.show_selection_handles = True
        self.handle_size = 8

        # 设置组件属性
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")

        # 启用鼠标跟踪
        self.setMouseTracking(True)

    def set_background_image(self, image: np.ndarray):
        """设置背景图像"""
        self.background_image = image.copy()
        self.update_canvas()

    def set_layer_manager(self, layer_manager: LayerManager):
        """设置图层管理器"""
        self.layer_manager = layer_manager
        self.update_canvas()

    def update_canvas(self):
        """更新画布显示"""
        if self.background_image is None:
            self.composite_image = None
            self.composite_pixmap = None
            self.update()
            return

        # 创建合成图像
        canvas_image = self.background_image.copy()

        if self.layer_manager:
            instances = self.layer_manager.get_all_instances()
            for i, instance in enumerate(instances):
                if instance.visible:
                    canvas_image = self._draw_instance_on_canvas(canvas_image, instance)

        # 保存合成图像
        self.composite_image = canvas_image

        # 转换为QPixmap
        qimage = numpy_to_qimage(canvas_image)
        if qimage:
            self.composite_pixmap = QPixmap.fromImage(qimage)
        else:
            print("QImage转换失败")
            self.composite_pixmap = None

        # 触发重绘
        self.update()

    def _draw_instance_on_canvas(
        self, canvas: np.ndarray, instance: MaterialInstance
    ) -> np.ndarray:
        """在画布上绘制素材实例"""
        try:
            transformed_image = instance.get_transformed_image()
            transformed_mask = instance.get_transformed_mask()

            x1, y1, x2, y2 = instance.get_bounding_rect()

            # 确保边界在画布范围内
            canvas_h, canvas_w = canvas.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(canvas_w, x2)
            y2 = min(canvas_h, y2)

            if x1 >= x2 or y1 >= y2:
                return canvas

            # 计算在变换图像中的对应区域
            orig_x1, orig_y1, _, _ = instance.get_bounding_rect()
            mat_x1 = max(0, -orig_x1) if orig_x1 < 0 else 0
            mat_y1 = max(0, -orig_y1) if orig_y1 < 0 else 0
            mat_x2 = mat_x1 + (x2 - x1)
            mat_y2 = mat_y1 + (y2 - y1)

            # 裁剪素材区域
            material_roi = transformed_image[mat_y1:mat_y2, mat_x1:mat_x2]

            if transformed_mask is not None:
                mask_roi = transformed_mask[mat_y1:mat_y2, mat_x1:mat_x2]

                # 改进掩码处理逻辑
                if len(mask_roi.shape) == 3:
                    # 如果掩码是彩色的，转换为灰度
                    gray_mask = cv2.cvtColor(mask_roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_mask = mask_roi.copy()

                # 创建二值掩码，使用更合适的阈值
                _, binary_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

                canvas_roi = canvas[y1:y2, x1:x2]

                # 检查尺寸匹配
                if (
                    material_roi.shape[:2] == canvas_roi.shape[:2]
                    and binary_mask.shape == material_roi.shape[:2]
                ):
                    # 根据混合模式选择合成方法
                    blend_mode = getattr(instance, "blend_mode", "normal")

                    if blend_mode == "poisson_normal":
                        # 泊松融合(正常模式)
                        canvas = self._poisson_blend(
                            canvas,
                            material_roi,
                            (x1, y1),
                            binary_mask,
                            cv2.NORMAL_CLONE,
                        )
                    elif blend_mode == "poisson_mixed":
                        # 泊松融合(混合模式)
                        canvas = self._poisson_blend(
                            canvas, material_roi, (x1, y1), binary_mask, cv2.MIXED_CLONE
                        )
                    else:
                        # 普通模式：简单的二值掩码混合
                        mask_3d = np.stack([binary_mask] * 3, axis=2) > 0
                        canvas_roi[mask_3d] = material_roi[mask_3d]
                else:
                    print(
                        f"尺寸不匹配: material_roi={material_roi.shape}, canvas_roi={canvas_roi.shape}, mask={binary_mask.shape}"
                    )
            else:
                # 直接覆盖（没有掩码的情况）
                canvas[y1:y2, x1:x2] = material_roi

        except Exception as e:
            print(f"绘制素材实例失败: {e}")
            import traceback

            traceback.print_exc()

        return canvas

    def _poisson_blend(
        self,
        dst: np.ndarray,
        src: np.ndarray,
        offset: Tuple[int, int],
        mask: np.ndarray,
        clone_type: int,
    ) -> np.ndarray:
        """执行泊松融合"""
        try:
            # 确保mask是单通道的
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # 创建掩码的中心点
            mask_center = (offset[0] + src.shape[1] // 2, offset[1] + src.shape[0] // 2)

            # 执行泊松融合
            result = cv2.seamlessClone(src, dst, mask, mask_center, clone_type)
            return result

        except Exception as e:
            print(f"泊松融合失败: {e}, 回退到普通混合")
            # 回退到普通混合模式
            try:
                x1, y1 = offset
                x2, y2 = x1 + src.shape[1], y1 + src.shape[0]

                # 确保边界在目标图像范围内
                dst_h, dst_w = dst.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(dst_w, x2)
                y2 = min(dst_h, y2)

                if x1 < x2 and y1 < y2:
                    # 调整源图像和掩码大小
                    adj_src = src[: y2 - y1, : x2 - x1]
                    adj_mask = mask[: y2 - y1, : x2 - x1]

                    # 简单混合
                    mask_3d = np.stack([adj_mask] * 3, axis=2) > 0
                    dst[y1:y2, x1:x2][mask_3d] = adj_src[mask_3d]

            except Exception as e2:
                print(f"回退混合也失败: {e2}")

            return dst

    def paintEvent(self, event: QPaintEvent):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        # 绘制合成图像
        if self.composite_pixmap:
            # 计算图像显示位置
            widget_rect = self.rect()

            # 计算缩放后的图像尺寸
            scaled_width = int(self.composite_pixmap.width() * self.zoom_factor)
            scaled_height = int(self.composite_pixmap.height() * self.zoom_factor)

            # 计算基础居中位置
            base_x = (widget_rect.width() - scaled_width) // 2
            base_y = (widget_rect.height() - scaled_height) // 2

            # 应用画布偏移
            image_x = base_x + self.canvas_offset.x()
            image_y = base_y + self.canvas_offset.y()

            # 绘制图像
            target_rect = QRect(image_x, image_y, scaled_width, scaled_height)
            painter.drawPixmap(target_rect, self.composite_pixmap)

        # 绘制边界框和选择控制点
        if self.show_bounding_boxes or self.show_selection_handles:
            if self.layer_manager:
                instances = self.layer_manager.get_all_instances()
                for instance in instances:
                    if instance.visible:
                        self._draw_instance_overlay(painter, instance)

        painter.end()

    def _draw_instance_overlay(self, painter: QPainter, instance: MaterialInstance):
        """绘制素材实例的叠加层（边界框等）"""
        # 使用mask边界框而不是图像边界框
        x1, y1, x2, y2 = instance.get_mask_bounding_rect()

        # 使用统一的坐标转换方法
        top_left_screen = self._canvas_to_screen_pos(QPoint(int(x1), int(y1)))
        bottom_right_screen = self._canvas_to_screen_pos(QPoint(int(x2), int(y2)))

        rect = QRect(
            top_left_screen.x(),
            top_left_screen.y(),
            bottom_right_screen.x() - top_left_screen.x(),
            bottom_right_screen.y() - top_left_screen.y(),
        )

        # 绘制边界框
        if instance == self.selected_instance:
            # 选中状态：绿色边框
            painter.setPen(QPen(QColor(0, 255, 0), 2))
        else:
            # 未选中状态：红色边框
            painter.setPen(QPen(QColor(255, 0, 0), 1))

        painter.setBrush(QBrush())  # 无填充
        painter.drawRect(rect)

        # 绘制选择控制点（仅选中时）
        if instance == self.selected_instance and self.show_selection_handles:
            self._draw_selection_handles(painter, rect)

    def _draw_selection_handles(self, painter: QPainter, rect: QRect):
        """绘制选择控制点"""
        handle_rects = self._get_handle_rects(rect)

        painter.setPen(QPen(QColor(0, 255, 0), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255)))

        for handle_rect in handle_rects:
            painter.drawRect(handle_rect)

    def _get_handle_rects(self, rect: QRect) -> List[QRect]:
        """获取控制点矩形列表"""
        handles = []
        half_size = self.handle_size // 2

        # 四个角的控制点
        handles.append(
            QRect(
                rect.left() - half_size,
                rect.top() - half_size,
                self.handle_size,
                self.handle_size,
            )
        )
        handles.append(
            QRect(
                rect.right() - half_size,
                rect.top() - half_size,
                self.handle_size,
                self.handle_size,
            )
        )
        handles.append(
            QRect(
                rect.left() - half_size,
                rect.bottom() - half_size,
                self.handle_size,
                self.handle_size,
            )
        )
        handles.append(
            QRect(
                rect.right() - half_size,
                rect.bottom() - half_size,
                self.handle_size,
                self.handle_size,
            )
        )

        # 四边中点的控制点
        handles.append(
            QRect(
                rect.center().x() - half_size,
                rect.top() - half_size,
                self.handle_size,
                self.handle_size,
            )
        )
        handles.append(
            QRect(
                rect.center().x() - half_size,
                rect.bottom() - half_size,
                self.handle_size,
                self.handle_size,
            )
        )
        handles.append(
            QRect(
                rect.left() - half_size,
                rect.center().y() - half_size,
                self.handle_size,
                self.handle_size,
            )
        )
        handles.append(
            QRect(
                rect.right() - half_size,
                rect.center().y() - half_size,
                self.handle_size,
                self.handle_size,
            )
        )

        return handles

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            canvas_pos = self._screen_to_canvas_pos(pos)

            # 查找点击的素材实例，增加容差提高选择精度
            clicked_instance = self._get_instance_at_pos_with_tolerance(
                canvas_pos.x(), canvas_pos.y(), tolerance=5
            )

            if clicked_instance:
                # 选中素材实例
                self.selected_instance = clicked_instance
                self.dragging = True
                self.drag_start_pos = pos
                self.drag_offset = QPoint(
                    canvas_pos.x() - clicked_instance.x,
                    canvas_pos.y() - clicked_instance.y,
                )
                self.instance_selected.emit(clicked_instance)
                # 不启动单击计时器，因为点击到了素材
                self.update()
            else:
                # 点击空白区域，启动双击检测
                self.pending_click_pos = canvas_pos
                self.click_timer.start(self.double_click_threshold)
                self.selected_instance = None
                self.instance_selected.emit(None)
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            # 右键始终拖动整个画布视图（背景+素材）
            pos = event.position().toPoint()
            self.canvas_dragging = True
            self.drag_start_pos = pos
            self.canvas_drag_start_offset = self.canvas_offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """鼠标双击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 停止单击计时器
            if self.click_timer.isActive():
                self.click_timer.stop()

            pos = event.position().toPoint()
            canvas_pos = self._screen_to_canvas_pos(pos)

            # 发送双击信号
            self.canvas_double_clicked.emit(canvas_pos.x(), canvas_pos.y())

    def _handle_single_click(self):
        """处理单击事件（延迟执行以区分双击）"""
        if self.pending_click_pos:
            # 发送单击信号
            self.canvas_clicked.emit(
                self.pending_click_pos.x(), self.pending_click_pos.y()
            )
            self.pending_click_pos = None

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if self.canvas_dragging:
            # 拖动整个画布视图 - 背景和所有素材一起移动
            current_pos = event.position().toPoint()
            delta = current_pos - self.drag_start_pos
            self.canvas_offset = self.canvas_drag_start_offset + delta
            # 只需要重绘显示，不需要重新合成图像
            self.update()
        elif self.dragging and self.selected_instance:
            # 拖动单个素材实例
            pos = event.position().toPoint()
            canvas_pos = self._screen_to_canvas_pos(pos)

            # 更新素材位置
            self.selected_instance.x = canvas_pos.x() - self.drag_offset.x()
            self.selected_instance.y = canvas_pos.y() - self.drag_offset.y()

            # 重新合成图像并更新显示
            self.update_canvas()
            self.instance_moved.emit(self.selected_instance)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.canvas_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件（缩放）"""
        delta = event.angleDelta().y()
        zoom_in = delta > 0

        # 获取鼠标在widget中的位置
        mouse_pos = event.position().toPoint()

        # 获取缩放前鼠标在画布坐标系中的位置
        old_canvas_pos = self._screen_to_canvas_pos(mouse_pos)

        # 更新缩放因子
        old_zoom = self.zoom_factor
        if zoom_in:
            self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.25)
        else:
            self.zoom_factor = max(self.min_zoom, self.zoom_factor / 1.25)

        # 只在缩放真的改变时调整偏移
        if old_zoom != self.zoom_factor:
            # 计算缩放后鼠标在画布坐标系中应该对应的新屏幕位置
            new_screen_pos = self._canvas_to_screen_pos(old_canvas_pos)

            # 计算偏移调整量，使鼠标位置保持不变
            offset_delta = mouse_pos - new_screen_pos
            self.canvas_offset += offset_delta

            # 更新显示
            self.update_canvas()

    def _screen_to_canvas_pos(self, screen_pos: QPoint) -> QPoint:
        """屏幕坐标转换为画布坐标"""
        if self.background_image is None:
            return screen_pos

        # 获取背景图像尺寸
        canvas_h, canvas_w = self.background_image.shape[:2]

        # 计算当前缩放后的图像尺寸
        scaled_width = int(canvas_w * self.zoom_factor)
        scaled_height = int(canvas_h * self.zoom_factor)

        # 计算图像在widget中的基础位置（居中对齐）
        widget_rect = self.rect()
        base_x = (widget_rect.width() - scaled_width) // 2
        base_y = (widget_rect.height() - scaled_height) // 2

        # 计算图像实际显示位置（考虑画布偏移）
        image_x_in_screen = base_x + self.canvas_offset.x()
        image_y_in_screen = base_y + self.canvas_offset.y()

        # 转换为原始图像坐标
        canvas_x = (screen_pos.x() - image_x_in_screen) / self.zoom_factor
        canvas_y = (screen_pos.y() - image_y_in_screen) / self.zoom_factor

        return QPoint(int(canvas_x), int(canvas_y))

    def _canvas_to_screen_pos(self, canvas_pos: QPoint) -> QPoint:
        """画布坐标转换为屏幕坐标"""
        if self.background_image is None:
            return canvas_pos

        # 获取背景图像尺寸
        canvas_h, canvas_w = self.background_image.shape[:2]

        # 计算当前缩放后的图像尺寸
        scaled_width = int(canvas_w * self.zoom_factor)
        scaled_height = int(canvas_h * self.zoom_factor)

        # 计算图像在widget中的基础位置（居中对齐）
        widget_rect = self.rect()
        base_x = (widget_rect.width() - scaled_width) // 2
        base_y = (widget_rect.height() - scaled_height) // 2

        # 计算图像实际显示位置（考虑画布偏移）
        image_x_in_screen = base_x + self.canvas_offset.x()
        image_y_in_screen = base_y + self.canvas_offset.y()

        # 转换为屏幕坐标
        screen_x = canvas_pos.x() * self.zoom_factor + image_x_in_screen
        screen_y = canvas_pos.y() * self.zoom_factor + image_y_in_screen

        return QPoint(int(screen_x), int(screen_y))

    def _get_instance_at_pos_with_tolerance(
        self, x: int, y: int, tolerance: int = 5
    ) -> Optional[MaterialInstance]:
        """获取指定位置的素材实例（带容差）"""
        if not self.layer_manager:
            return None

        # 先尝试精确匹配
        instances = self.layer_manager.get_instances_at_point(x, y)
        if instances:
            return instances[0]

        # 如果没有精确匹配，尝试容差范围内的匹配
        for dx in range(-tolerance, tolerance + 1):
            for dy in range(-tolerance, tolerance + 1):
                if dx == 0 and dy == 0:
                    continue
                instances = self.layer_manager.get_instances_at_point(x + dx, y + dy)
                if instances:
                    return instances[0]

        return None

    def zoom_to_fit(self):
        """缩放到适合窗口大小"""
        if self.background_image is None:
            return

        canvas_h, canvas_w = self.background_image.shape[:2]
        widget_size = self.size()

        scale_w = widget_size.width() / canvas_w
        scale_h = widget_size.height() / canvas_h

        self.zoom_factor = min(scale_w, scale_h, 1.0)
        # 重置画布偏移
        self.canvas_offset = QPoint(0, 0)
        # 只需要重绘，不需要重新合成
        self.update()

    def zoom_to_actual_size(self):
        """缩放到实际大小"""
        self.zoom_factor = 1.0
        # 重置画布偏移
        self.canvas_offset = QPoint(0, 0)
        # 只需要重绘，不需要重新合成
        self.update()

    def add_material_at_pos(self, material_name: str, x: int, y: int) -> bool:
        """在指定位置添加素材"""
        if not self.layer_manager:
            return False

        from core.material import MaterialManager

        # 这里需要传入MaterialManager实例，暂时返回False
        # 实际使用时需要从主窗口获取MaterialManager
        return False


class CanvasScrollArea(QScrollArea):
    """带滚动条的画布区域"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.canvas = CanvasWidget()
        self.setWidget(self.canvas)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 设置滚动条策略
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def get_canvas(self) -> CanvasWidget:
        """获取画布组件"""
        return self.canvas
