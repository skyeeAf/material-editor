"""
优化版画布组件
修复泊松融合性能问题，增加智能缓存和渲染优化
"""

import math
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

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


class PoissonRenderCache:
    """泊松融合渲染缓存"""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._access_order: List[str] = []

    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存的渲染结果"""
        if key in self._cache:
            result, timestamp = self._cache[key]
            # 检查缓存是否过期（5秒）
            if time.time() - timestamp < 5.0:
                # 更新访问顺序
                self._access_order.remove(key)
                self._access_order.append(key)
                return result.copy()
            else:
                # 过期，删除缓存
                del self._cache[key]
                self._access_order.remove(key)

        return None

    def set(self, key: str, result: np.ndarray):
        """设置缓存"""
        # 如果缓存已满，删除最旧的项
        if len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

        # 添加新缓存
        self._cache[key] = (result.copy(), time.time())
        self._access_order.append(key)


class OptimizedCanvasWidget(QLabel):
    """优化版画布组件"""

    # 信号定义
    instance_selected = Signal(object)
    instance_moved = Signal(object)
    instance_transformed = Signal(object)
    canvas_clicked = Signal(int, int)
    canvas_double_clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 画布属性
        self.background_image: Optional[np.ndarray] = None
        self.layer_manager: Optional[LayerManager] = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # 画布视口偏移
        self.canvas_offset = QPoint(0, 0)

        # 渲染缓存系统
        self.composite_image: Optional[np.ndarray] = None
        self.composite_pixmap: Optional[QPixmap] = None
        self.poisson_cache = PoissonRenderCache(max_size=30)

        # 分层渲染缓存
        self.background_with_normal_cache: Optional[np.ndarray] = None
        self.normal_instances_hash: Optional[str] = None

        # 性能优化标志
        self.enable_smart_render = True
        self.enable_poisson_cache = True
        self.max_poisson_items_per_frame = 3  # 每帧最多处理的泊松融合项目数

        # 交互状态
        self.selected_instance: Optional[MaterialInstance] = None
        self.dragging = False
        self.canvas_dragging = False
        self.drag_start_pos = QPoint()
        self.drag_offset = QPoint()
        self.canvas_drag_start_offset = QPoint()

        # 智能更新系统
        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._smart_render)
        self.pending_render = False

        # 拖动优化 - 降低更新频率
        self.drag_update_timer = QTimer()
        self.drag_update_timer.setSingleShot(True)
        self.drag_update_timer.timeout.connect(self._delayed_canvas_update)
        self.pending_drag_update = False

        # 双击检测
        self.click_timer = QTimer()
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self._handle_single_click)
        self.pending_click_pos = None

        # 视觉效果
        self.show_bounding_boxes = True
        self.show_selection_handles = True
        self.handle_size = 8

        # 性能统计
        self.render_stats = {
            "last_render_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "poisson_operations": 0,
        }

        # 设置组件属性
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.setMouseTracking(True)

    def set_background_image(self, image: np.ndarray):
        """设置背景图像"""
        self.background_image = image.copy()
        self._invalidate_all_caches()
        self._schedule_render()

    def set_layer_manager(self, layer_manager: LayerManager):
        """设置图层管理器"""
        self.layer_manager = layer_manager
        self._invalidate_all_caches()
        self._schedule_render()

    def _invalidate_all_caches(self):
        """使所有缓存失效"""
        self.background_with_normal_cache = None
        self.normal_instances_hash = None
        self.poisson_cache._cache.clear()
        self.poisson_cache._access_order.clear()

    def _schedule_render(self):
        """安排渲染"""
        if not self.render_timer.isActive():
            self.render_timer.start(16)  # 约60FPS

    def _smart_render(self):
        """智能渲染"""
        start_time = time.time()

        if self.background_image is None:
            self.composite_image = None
            self.composite_pixmap = None
            self.update()
            return

        if not self.layer_manager:
            self.composite_image = self.background_image.copy()
        else:
            self.composite_image = self._render_with_smart_cache()

        # 转换为QPixmap
        if self.composite_image is not None:
            qimage = numpy_to_qimage(self.composite_image)
            if qimage:
                self.composite_pixmap = QPixmap.fromImage(qimage)
            else:
                self.composite_pixmap = None

        # 更新性能统计
        self.render_stats["last_render_time"] = time.time() - start_time

        # 触发重绘
        self.update()

    def _render_with_smart_cache(self) -> np.ndarray:
        """使用智能缓存进行渲染"""
        if not self.layer_manager:
            return self.background_image.copy()

        instances = self.layer_manager.get_all_instances()
        if not instances:
            return self.background_image.copy()

        # 分类实例
        normal_instances = [
            i for i in instances if i.visible and i.blend_mode == "normal"
        ]
        poisson_instances = [
            i
            for i in instances
            if i.visible and i.blend_mode in ["poisson_normal", "poisson_mixed"]
        ]

        # 第一步：渲染普通混合模式实例（可缓存）
        canvas = self._render_normal_instances_cached(normal_instances)

        # 第二步：渲染泊松融合实例（使用缓存）
        canvas = self._render_poisson_instances_cached(canvas, poisson_instances)

        return canvas

    def _generate_normal_instances_hash(self, instances: List[MaterialInstance]) -> str:
        """生成普通实例的哈希值"""
        if not instances:
            return "empty"

        hash_parts = []
        for instance in sorted(instances, key=lambda x: id(x)):
            hash_parts.append(
                f"{id(instance)}_{instance.x}_{instance.y}_{instance.scale}_{instance.rotation}_{instance.color_overlay}_{instance.overlay_opacity}"
            )

        return hash("_".join(hash_parts))

    def _render_normal_instances_cached(
        self, instances: List[MaterialInstance]
    ) -> np.ndarray:
        """使用缓存渲染普通混合模式实例"""
        current_hash = self._generate_normal_instances_hash(instances)

        # 检查缓存
        if (
            self.background_with_normal_cache is not None
            and self.normal_instances_hash == current_hash
        ):
            return self.background_with_normal_cache.copy()

        # 重新渲染
        canvas = self.background_image.copy()

        for instance in instances:
            canvas = self._render_normal_instance(canvas, instance)

        # 更新缓存
        self.background_with_normal_cache = canvas.copy()
        self.normal_instances_hash = current_hash

        return canvas

    def _render_poisson_instances_cached(
        self, canvas: np.ndarray, instances: List[MaterialInstance]
    ) -> np.ndarray:
        """使用缓存渲染泊松融合实例"""
        if not instances:
            return canvas

        # 限制每帧处理的泊松融合数量
        processed_count = 0

        for instance in instances:
            if processed_count >= self.max_poisson_items_per_frame:
                break  # 延迟到下一帧处理

            cache_key = self._generate_poisson_cache_key(instance)

            if self.enable_poisson_cache:
                cached_result = self.poisson_cache.get(cache_key)
                if cached_result is not None:
                    # 缓存命中，直接应用结果
                    canvas = self._apply_cached_poisson_result(
                        canvas, cached_result, instance
                    )
                    self.render_stats["cache_hits"] += 1
                    continue

            # 缓存未命中，执行泊松融合
            canvas = self._render_poisson_instance_with_cache(
                canvas, instance, cache_key
            )
            self.render_stats["cache_misses"] += 1
            self.render_stats["poisson_operations"] += 1
            processed_count += 1

        return canvas

    def _generate_poisson_cache_key(self, instance: MaterialInstance) -> str:
        """生成泊松融合缓存键"""
        return f"poisson_{id(instance)}_{instance.x}_{instance.y}_{instance.scale}_{instance.rotation}_{instance.blend_mode}"

    def _render_poisson_instance_with_cache(
        self, canvas: np.ndarray, instance: MaterialInstance, cache_key: str
    ) -> np.ndarray:
        """渲染泊松融合实例并缓存结果"""
        try:
            original_canvas = canvas.copy()
            result_canvas = self._render_poisson_instance(canvas, instance)

            # 缓存差异而不是整个图像
            if self.enable_poisson_cache:
                diff = result_canvas - original_canvas
                self.poisson_cache.set(cache_key, diff)

            return result_canvas

        except Exception as e:
            print(f"泊松融合渲染失败: {e}")
            return self._render_normal_instance(canvas, instance)

    def _apply_cached_poisson_result(
        self, canvas: np.ndarray, cached_diff: np.ndarray, instance: MaterialInstance
    ) -> np.ndarray:
        """应用缓存的泊松融合结果"""
        try:
            return canvas + cached_diff
        except:
            # 如果应用缓存失败，重新渲染
            return self._render_poisson_instance(canvas, instance)

    def _render_normal_instance(
        self, canvas: np.ndarray, instance: MaterialInstance
    ) -> np.ndarray:
        """渲染普通混合模式实例"""
        try:
            transformed_image = instance.get_transformed_image()
            transformed_mask = instance.get_transformed_mask()

            if transformed_image is None or transformed_mask is None:
                return canvas

            # 计算位置
            h, w = transformed_image.shape[:2]
            canvas_h, canvas_w = canvas.shape[:2]

            center_x = int(instance.x)
            center_y = int(instance.y)
            x = center_x - w // 2
            y = center_y - h // 2

            # 计算有效区域
            src_x1 = max(0, -x)
            src_y1 = max(0, -y)
            src_x2 = min(w, canvas_w - x)
            src_y2 = min(h, canvas_h - y)

            dst_x1 = max(0, x)
            dst_y1 = max(0, y)
            dst_x2 = min(canvas_w, x + w)
            dst_y2 = min(canvas_h, y + h)

            # 检查有效性
            if (
                src_x2 <= src_x1
                or src_y2 <= src_y1
                or dst_x2 <= dst_x1
                or dst_y2 <= dst_y1
            ):
                return canvas

            # 提取区域
            src_image = transformed_image[src_y1:src_y2, src_x1:src_x2]
            src_mask = transformed_mask[src_y1:src_y2, src_x1:src_x2]

            # 混合
            canvas_roi = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
            mask_3d = src_mask.astype(np.float32) / 255.0
            if len(mask_3d.shape) == 2:
                mask_3d = np.stack([mask_3d] * 3, axis=2)

            blended_roi = (
                canvas_roi.astype(np.float32) * (1 - mask_3d)
                + src_image.astype(np.float32) * mask_3d
            )

            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = blended_roi.astype(np.uint8)

        except Exception as e:
            print(f"普通混合渲染失败: {e}")

        return canvas

    def _render_poisson_instance(
        self, canvas: np.ndarray, instance: MaterialInstance
    ) -> np.ndarray:
        """渲染泊松融合实例"""
        try:
            transformed_image = instance.get_transformed_image()
            transformed_mask = instance.get_transformed_mask()

            if transformed_image is None or transformed_mask is None:
                return canvas

            # 执行泊松融合
            clone_type = (
                cv2.NORMAL_CLONE
                if instance.blend_mode == "poisson_normal"
                else cv2.MIXED_CLONE
            )

            # 确保mask是单通道
            if len(transformed_mask.shape) == 3:
                mask = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY)
            else:
                mask = transformed_mask

            mask_center = (int(instance.x), int(instance.y))
            result = cv2.seamlessClone(
                transformed_image, canvas, mask, mask_center, clone_type
            )

            return result

        except Exception as e:
            print(f"泊松融合失败: {e}")
            # 回退到普通混合
            return self._render_normal_instance(canvas, instance)

    def update_canvas(self, force_full_update: bool = False):
        """更新画布显示"""
        if force_full_update:
            self._invalidate_all_caches()

        self._schedule_render()

    # 以下是交互事件处理方法...
    def paintEvent(self, event: QPaintEvent):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        # 绘制合成图像
        if self.composite_pixmap:
            widget_rect = self.rect()
            scaled_width = int(self.composite_pixmap.width() * self.zoom_factor)
            scaled_height = int(self.composite_pixmap.height() * self.zoom_factor)

            base_x = (widget_rect.width() - scaled_width) // 2
            base_y = (widget_rect.height() - scaled_height) // 2

            image_x = base_x + self.canvas_offset.x()
            image_y = base_y + self.canvas_offset.y()

            target_rect = QRect(image_x, image_y, scaled_width, scaled_height)
            painter.drawPixmap(target_rect, self.composite_pixmap)

        # 绘制UI叠加层
        if self.show_bounding_boxes and self.layer_manager:
            instances = self.layer_manager.get_all_instances()
            for instance in instances:
                if instance.visible:
                    self._draw_instance_overlay(painter, instance)

        painter.end()

    def _draw_instance_overlay(self, painter: QPainter, instance: MaterialInstance):
        """绘制实例叠加层"""
        x1, y1, x2, y2 = instance.get_mask_bounding_rect()

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
            painter.setPen(QPen(QColor(0, 255, 0), 2))
        else:
            painter.setPen(QPen(QColor(255, 0, 0), 1))

        painter.setBrush(QBrush())
        painter.drawRect(rect)

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            canvas_pos = self._screen_to_canvas_pos(event.position().toPoint())

            # 查找点击的实例
            if self.layer_manager:
                instances = self.layer_manager.get_instances_at_point(
                    canvas_pos.x(), canvas_pos.y()
                )
                if instances:
                    self.selected_instance = instances[0]
                    self.dragging = True

                    # 计算拖动偏移
                    self.drag_offset = QPoint(
                        canvas_pos.x() - self.selected_instance.x,
                        canvas_pos.y() - self.selected_instance.y,
                    )

                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self.instance_selected.emit(self.selected_instance)
                    return

        elif event.button() == Qt.MouseButton.RightButton:
            # 右键拖动画布
            self.canvas_dragging = True
            self.drag_start_pos = event.position().toPoint()
            self.canvas_drag_start_offset = self.canvas_offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        pos = event.position().toPoint()

        if self.dragging and self.selected_instance:
            # 拖拽素材实例
            canvas_pos = self._screen_to_canvas_pos(pos)
            self.selected_instance.x = canvas_pos.x() - self.drag_offset.x()
            self.selected_instance.y = canvas_pos.y() - self.drag_offset.y()

            # 使拖动时的缓存失效
            self._invalidate_all_caches()

            # 节流更新
            if not self.pending_drag_update:
                self.pending_drag_update = True
                self.drag_update_timer.start(32)  # 约30FPS拖动更新

        elif self.canvas_dragging:
            # 拖拽画布
            delta = pos - self.drag_start_pos
            self.canvas_offset = self.canvas_drag_start_offset + delta
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            self.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

            # 拖动结束后完全重新渲染
            self.update_canvas(force_full_update=True)

            if self.selected_instance:
                self.instance_moved.emit(self.selected_instance)

        elif event.button() == Qt.MouseButton.RightButton and self.canvas_dragging:
            self.canvas_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _delayed_canvas_update(self):
        """延迟的画布更新"""
        if self.pending_drag_update:
            self._schedule_render()
            self.pending_drag_update = False

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        delta = event.angleDelta().y()
        zoom_in = delta > 0

        mouse_pos = event.position().toPoint()
        old_canvas_pos = self._screen_to_canvas_pos(mouse_pos)

        old_zoom = self.zoom_factor
        if zoom_in:
            self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.25)
        else:
            self.zoom_factor = max(self.min_zoom, self.zoom_factor / 1.25)

        if old_zoom != self.zoom_factor:
            new_screen_pos = self._canvas_to_screen_pos(old_canvas_pos)
            offset_delta = mouse_pos - new_screen_pos
            self.canvas_offset += offset_delta
            self.update()

    def _screen_to_canvas_pos(self, screen_pos: QPoint) -> QPoint:
        """屏幕坐标转画布坐标"""
        if self.background_image is None:
            return screen_pos

        canvas_h, canvas_w = self.background_image.shape[:2]
        scaled_width = int(canvas_w * self.zoom_factor)
        scaled_height = int(canvas_h * self.zoom_factor)

        widget_rect = self.rect()
        base_x = (widget_rect.width() - scaled_width) // 2
        base_y = (widget_rect.height() - scaled_height) // 2

        image_x_in_screen = base_x + self.canvas_offset.x()
        image_y_in_screen = base_y + self.canvas_offset.y()

        canvas_x = (screen_pos.x() - image_x_in_screen) / self.zoom_factor
        canvas_y = (screen_pos.y() - image_y_in_screen) / self.zoom_factor

        return QPoint(int(canvas_x), int(canvas_y))

    def _canvas_to_screen_pos(self, canvas_pos: QPoint) -> QPoint:
        """画布坐标转屏幕坐标"""
        if self.background_image is None:
            return canvas_pos

        canvas_h, canvas_w = self.background_image.shape[:2]
        scaled_width = int(canvas_w * self.zoom_factor)
        scaled_height = int(canvas_h * self.zoom_factor)

        widget_rect = self.rect()
        base_x = (widget_rect.width() - scaled_width) // 2
        base_y = (widget_rect.height() - scaled_height) // 2

        image_x_in_screen = base_x + self.canvas_offset.x()
        image_y_in_screen = base_y + self.canvas_offset.y()

        screen_x = canvas_pos.x() * self.zoom_factor + image_x_in_screen
        screen_y = canvas_pos.y() * self.zoom_factor + image_y_in_screen

        return QPoint(int(screen_x), int(screen_y))

    def _handle_single_click(self):
        """处理单击事件"""
        if self.pending_click_pos:
            self.canvas_clicked.emit(
                self.pending_click_pos.x(), self.pending_click_pos.y()
            )
            self.pending_click_pos = None

    def zoom_to_fit(self):
        """缩放到适合窗口"""
        if self.background_image is None:
            return

        canvas_h, canvas_w = self.background_image.shape[:2]
        widget_size = self.size()

        scale_w = widget_size.width() / canvas_w
        scale_h = widget_size.height() / canvas_h

        self.zoom_factor = min(scale_w, scale_h, 1.0)
        self.canvas_offset = QPoint(0, 0)
        self.update()

    def zoom_to_actual_size(self):
        """缩放到实际大小"""
        self.zoom_factor = 1.0
        self.canvas_offset = QPoint(0, 0)
        self.update()

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            **self.render_stats,
            "cache_efficiency": (
                self.render_stats["cache_hits"]
                / max(
                    1,
                    self.render_stats["cache_hits"] + self.render_stats["cache_misses"],
                )
            ),
            "cached_items": len(self.poisson_cache._cache),
        }


class OptimizedCanvasScrollArea(QScrollArea):
    """优化版画布滚动区域"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建优化版画布
        self.canvas = OptimizedCanvasWidget()
        self.setWidget(self.canvas)

        # 设置滚动区域属性
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def get_canvas(self) -> OptimizedCanvasWidget:
        """获取画布组件"""
        return self.canvas
