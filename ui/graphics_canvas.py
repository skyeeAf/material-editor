"""
高性能图形画布组件
基于QGraphicsView实现，针对泊松融合场景优化
"""

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

from core.layer import LayerManager
from core.material import MaterialInstance
from utils.image_utils import numpy_to_qimage


@dataclass
class RenderCache:
    """渲染缓存"""

    key: str
    pixmap: QPixmap
    timestamp: float

    def is_valid(self, max_age: float = 5.0) -> bool:
        """检查缓存是否有效"""
        return time.time() - self.timestamp < max_age


class MaterialGraphicsItem(QGraphicsItem):
    """素材图形项"""

    def __init__(self, instance: MaterialInstance, parent=None):
        super().__init__(parent)
        self.instance = instance
        self._cached_pixmap: Optional[QPixmap] = None
        self._cache_key: Optional[str] = None
        self._bounding_rect: Optional[QRectF] = None
        self._graphics_view: Optional[HighPerformanceGraphicsView] = None

        # 设置图形项属性
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # 设置位置
        self.setPos(instance.x, instance.y)

        # 不在这里设置变换，因为变换已经应用到图像上了
        # MaterialInstance.get_transformed_image()已经包含了缩放和旋转

    def set_graphics_view(self, graphics_view):
        """设置图形视图引用"""
        self._graphics_view = graphics_view

    def boundingRect(self) -> QRectF:
        """返回边界矩形 - 使用mask边界"""
        if self._bounding_rect is None:
            self._update_bounding_rect()
        return self._bounding_rect

    def _update_bounding_rect(self):
        """更新边界矩形 - 基于mask边界"""
        try:
            # 检查是否为泊松融合实例
            is_poisson = self.instance.blend_mode in ["poisson_normal", "poisson_mixed"]

            # 获取变换后的图像和mask
            transformed_image = self.instance.get_transformed_image()
            transformed_mask = self.instance.get_transformed_mask()

            if transformed_image is None:
                self._bounding_rect = QRectF()
                return

            # 计算mask的有效区域
            if transformed_mask is not None:
                # 转换为灰度
                if len(transformed_mask.shape) == 3:
                    gray_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY)
                else:
                    gray_mask = transformed_mask

                # 找到非零区域
                coords = np.column_stack(np.where(gray_mask > 10))
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)

                    # 计算mask的有效区域尺寸
                    width = max(1, x_max - x_min + 1)  # 确保宽度至少为1
                    height = max(1, y_max - y_min + 1)  # 确保高度至少为1

                    if is_poisson:
                        # 泊松融合实例：边界矩形应该相对于实例的中心点(0,0)
                        # 因为泊松融合的输入点是中心点，所以这里的坐标应该相对于中心
                        img_h, img_w = transformed_image.shape[:2]
                        img_center_x, img_center_y = img_w // 2, img_h // 2

                        # 计算mask边界相对于图像中心的偏移
                        mask_center_x = (x_min + x_max) // 2
                        mask_center_y = (y_min + y_max) // 2

                        # 计算边界矩形相对于实例中心点的位置
                        rect_center_x = mask_center_x - img_center_x
                        rect_center_y = mask_center_y - img_center_y

                        # 边界矩形的左上角相对于中心的偏移
                        left = rect_center_x - width // 2
                        top = rect_center_y - height // 2

                        self._bounding_rect = QRectF(left, top, width, height)

                        print(
                            f"泊松融合实例 {self.instance.material_name} mask边界矩形: {self._bounding_rect}"
                        )
                        print(
                            f"  图像尺寸: {img_w}x{img_h}, 图像中心: ({img_center_x}, {img_center_y})"
                        )
                        print(f"  mask范围: x[{x_min}, {x_max}], y[{y_min}, {y_max}]")
                        print(f"  mask中心: ({mask_center_x}, {mask_center_y})")
                        print(f"  相对中心偏移: ({rect_center_x}, {rect_center_y})")
                    else:
                        # 普通实例：使用原来的逻辑
                        h, w = transformed_image.shape[:2]
                        if h > 0 and w > 0:
                            center_x, center_y = w // 2, h // 2
                            left = x_min - center_x
                            top = y_min - center_y
                            self._bounding_rect = QRectF(left, top, width, height)
                        else:
                            self._bounding_rect = QRectF(-1, -1, 2, 2)
                else:
                    # 如果mask为空，使用完整图像边界
                    h, w = transformed_image.shape[:2]
                    if h > 0 and w > 0:
                        self._bounding_rect = QRectF(-w / 2, -h / 2, w, h)
                    else:
                        self._bounding_rect = QRectF(-1, -1, 2, 2)
            else:
                # 没有mask，使用完整图像边界
                h, w = transformed_image.shape[:2]
                if h > 0 and w > 0:
                    self._bounding_rect = QRectF(-w / 2, -h / 2, w, h)
                else:
                    self._bounding_rect = QRectF(-1, -1, 2, 2)

        except Exception as e:
            print(f"更新边界矩形失败: {e}")
            # 设置默认边界矩形
            self._bounding_rect = QRectF(-50, -50, 100, 100)

    def paint(self, painter: QPainter, option, widget):
        """绘制项目"""
        try:
            # 检查是否为泊松融合实例
            is_poisson = self.instance.blend_mode in ["poisson_normal", "poisson_mixed"]

            if is_poisson:
                # 泊松融合实例：绘制轮廓用于交互
                try:
                    # 获取mask边界矩形（已经考虑了偏移）
                    bounding_rect = self.boundingRect()
                    if not bounding_rect.isEmpty():
                        # 绘制主边框轮廓
                        painter.setPen(QPen(QColor(0, 255, 255, 200), 2))  # 青色边框
                        painter.setBrush(QBrush())  # 无填充
                        painter.drawRect(bounding_rect)

                        # 在四个角绘制小方块作为选择指示
                        corner_size = 6
                        painter.setBrush(QBrush(QColor(0, 255, 255, 200)))

                        # 左上角
                        painter.drawRect(
                            bounding_rect.left() - corner_size / 2,
                            bounding_rect.top() - corner_size / 2,
                            corner_size,
                            corner_size,
                        )
                        # 右上角
                        painter.drawRect(
                            bounding_rect.right() - corner_size / 2,
                            bounding_rect.top() - corner_size / 2,
                            corner_size,
                            corner_size,
                        )
                        # 左下角
                        painter.drawRect(
                            bounding_rect.left() - corner_size / 2,
                            bounding_rect.bottom() - corner_size / 2,
                            corner_size,
                            corner_size,
                        )
                        # 右下角
                        painter.drawRect(
                            bounding_rect.right() - corner_size / 2,
                            bounding_rect.bottom() - corner_size / 2,
                            corner_size,
                            corner_size,
                        )

                except Exception as e:
                    print(f"绘制泊松融合轮廓失败: {e}")
            else:
                # 普通实例：正常绘制
                pixmap = self._get_cached_pixmap()
                if pixmap and not pixmap.isNull():
                    # 获取图像的完整矩形
                    h, w = self._get_image_size()
                    if h > 0 and w > 0:
                        full_rect = QRectF(-w / 2, -h / 2, w, h)
                        painter.drawPixmap(full_rect, pixmap, QRectF(pixmap.rect()))

            # 绘制选择框
            if self.isSelected():
                try:
                    painter.setPen(QPen(QColor(0, 255, 0), 3))  # 增加选择框宽度
                    painter.setBrush(QBrush())
                    bounding_rect = self.boundingRect()
                    if not bounding_rect.isEmpty():
                        painter.drawRect(bounding_rect)
                except Exception as e:
                    print(f"绘制选择框失败: {e}")

        except Exception as e:
            print(f"绘制MaterialGraphicsItem失败: {e}")
            # 绘制一个简单的错误指示矩形
            try:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawRect(-25, -25, 50, 50)
                painter.drawText(-20, 0, "ERROR")
            except:
                pass

    def _get_image_size(self) -> Tuple[int, int]:
        """获取图像尺寸"""
        transformed_image = self.instance.get_transformed_image()
        if transformed_image is not None:
            return transformed_image.shape[:2]
        return (0, 0)

    def _get_masked_image(self) -> Optional[np.ndarray]:
        """获取应用mask后的图像"""
        try:
            transformed_image = self.instance.get_transformed_image()
            transformed_mask = self.instance.get_transformed_mask()

            if transformed_image is None:
                return None

            if transformed_mask is None:
                return transformed_image

            # 确保图像和mask尺寸匹配
            img_h, img_w = transformed_image.shape[:2]
            mask_h, mask_w = transformed_mask.shape[:2]

            if img_h != mask_h or img_w != mask_w:
                print(
                    f"警告：图像尺寸({img_w}x{img_h})与mask尺寸({mask_w}x{mask_h})不匹配"
                )
                # 调整mask尺寸以匹配图像
                transformed_mask = cv2.resize(transformed_mask, (img_w, img_h))

            # 应用mask
            if len(transformed_mask.shape) == 3:
                mask_gray = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = transformed_mask

            # 创建mask后的图像
            if len(transformed_image.shape) == 3:
                # BGR图像，创建BGRA
                h, w = transformed_image.shape[:2]
                masked_image = np.zeros((h, w, 4), dtype=np.uint8)
                masked_image[:, :, :3] = transformed_image  # 复制BGR通道
                masked_image[:, :, 3] = mask_gray  # 设置alpha通道
            else:
                # 灰度图像，创建RGBA
                h, w = transformed_image.shape[:2]
                masked_image = np.zeros((h, w, 4), dtype=np.uint8)
                # 将灰度图转换为BGR再复制
                bgr_image = cv2.cvtColor(transformed_image, cv2.COLOR_GRAY2BGR)
                masked_image[:, :, :3] = bgr_image
                masked_image[:, :, 3] = mask_gray

            return masked_image

        except Exception as e:
            print(f"获取mask后图像失败: {e}")
            # 返回原始图像作为回退
            try:
                transformed_image = self.instance.get_transformed_image()
                if transformed_image is not None and len(transformed_image.shape) == 3:
                    # 如果是BGR图像，转换为BGRA以保持一致性
                    h, w = transformed_image.shape[:2]
                    bgra_image = np.zeros((h, w, 4), dtype=np.uint8)
                    bgra_image[:, :, :3] = transformed_image
                    bgra_image[:, :, 3] = 255  # 完全不透明
                    return bgra_image
                return transformed_image
            except:
                return None

    def _get_cached_pixmap(self) -> Optional[QPixmap]:
        """获取缓存的像素图"""
        cache_key = self._generate_cache_key()

        if self._cache_key != cache_key or self._cached_pixmap is None:
            self._cached_pixmap = self._create_pixmap()
            self._cache_key = cache_key
            # 缓存更新时也更新边界矩形
            self._bounding_rect = None

        return self._cached_pixmap

    def _generate_cache_key(self) -> str:
        """生成缓存键"""
        return f"{id(self.instance)}_{self.instance.scale}_{self.instance.rotation}_{self.instance.color_overlay}_{self.instance.overlay_opacity}_{self.instance.x}_{self.instance.y}"

    def _create_pixmap(self) -> Optional[QPixmap]:
        """创建像素图 - 使用mask后的图像"""
        try:
            masked_image = self._get_masked_image()
            if masked_image is None:
                return None

            # 检查图像有效性
            if masked_image.size == 0:
                print("警告：图像为空")
                return None

            h, w = masked_image.shape[:2]
            if h <= 0 or w <= 0:
                print(f"警告：图像尺寸无效 {w}x{h}")
                return None

            qimage = numpy_to_qimage(masked_image)
            if qimage and not qimage.isNull():
                pixmap = QPixmap.fromImage(qimage)
                if not pixmap.isNull():
                    return pixmap
                else:
                    print("警告：QPixmap创建失败")
            else:
                print("警告：QImage创建失败")

        except Exception as e:
            print(f"创建像素图失败: {e}")
            import traceback

            traceback.print_exc()

        return None

    def update_from_instance(self):
        """从实例更新显示"""
        try:
            # 清除所有缓存，强制重新生成
            self._cached_pixmap = None
            self._cache_key = None

            # 清除实例的变换缓存（特别是颜色叠加变化时）
            if hasattr(self.instance, "_transformed_image_cache"):
                self.instance._transformed_image_cache = None
            if hasattr(self.instance, "_transform_cache_key"):
                self.instance._transform_cache_key = None

            # 更新边界矩形
            self._update_bounding_rect()

            # 更新图形项位置
            self.setPos(self.instance.x, self.instance.y)

            # 标记需要重绘
            self.update()

            print(
                f"MaterialGraphicsItem更新完成: {self.instance.material_name}, 颜色叠加: {self.instance.color_overlay}, 透明度: {self.instance.overlay_opacity}"
            )

        except Exception as e:
            print(f"MaterialGraphicsItem更新失败: {e}")
            import traceback

            traceback.print_exc()

    def itemChange(self, change, value):
        """项目变化处理"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # 更新实例位置
            pos = value
            self.instance.x = int(pos.x())
            self.instance.y = int(pos.y())

            # 发送移动信号
            if self._graphics_view:
                self._graphics_view.instance_moved.emit(self.instance)
                # 标记需要重新合成
                self._graphics_view.composite_dirty = True
                self._graphics_view._schedule_update()

        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # 选择状态改变
            if value and self._graphics_view:
                self._graphics_view.instance_selected.emit(self.instance)

        return super().itemChange(change, value)


class PoissonBlendCache:
    """泊松融合缓存"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, RenderCache] = {}
        self._access_order: List[str] = []

    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存"""
        if key in self._cache:
            cache_item = self._cache[key]
            if cache_item.is_valid():
                # 更新访问顺序
                self._access_order.remove(key)
                self._access_order.append(key)

                # 从QPixmap转换回numpy数组 - 修复形状问题
                try:
                    qimage = cache_item.pixmap.toImage()
                    # 确保格式正确
                    if qimage.format() != qimage.Format.Format_RGB888:
                        qimage = qimage.convertToFormat(qimage.Format.Format_RGB888)

                    width, height = qimage.width(), qimage.height()
                    # 检查图像有效性
                    if width <= 0 or height <= 0:
                        print(f"警告：缓存图像尺寸无效 {width}x{height}")
                        return None

                    ptr = qimage.constBits()
                    # 计算预期的数组大小
                    expected_size = height * width * 3

                    try:
                        arr = np.frombuffer(ptr, dtype=np.uint8, count=expected_size)
                        arr = arr.reshape((height, width, 3))
                        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    except ValueError as e:
                        print(
                            f"数组重塑失败: {e}, 图像尺寸: {width}x{height}, 预期大小: {expected_size}"
                        )
                        return None

                except Exception as e:
                    print(f"从缓存获取图像失败: {e}")
                    return None

        return None

    def set(self, key: str, image: np.ndarray):
        """设置缓存"""
        try:
            # 清理过期缓存
            self._cleanup_expired()

            # 如果缓存已满，删除最旧的
            if len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]

            # 检查图像有效性
            if image is None or image.size == 0:
                print("警告：尝试缓存空图像")
                return

            # 转换为QPixmap并缓存
            qimage = numpy_to_qimage(image)
            if qimage and not qimage.isNull():
                pixmap = QPixmap.fromImage(qimage)
                if not pixmap.isNull():
                    cache_item = RenderCache(key, pixmap, time.time())
                    self._cache[key] = cache_item
                    self._access_order.append(key)
                else:
                    print("警告：QPixmap创建失败")
            else:
                print("警告：QImage创建失败")

        except Exception as e:
            print(f"设置缓存失败: {e}")

    def _cleanup_expired(self):
        """清理过期缓存"""
        expired_keys = [
            key for key, cache_item in self._cache.items() if not cache_item.is_valid()
        ]

        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)


class HighPerformanceGraphicsView(QGraphicsView):
    """高性能图形视图"""

    # 信号定义
    instance_selected = Signal(object)
    instance_moved = Signal(object)
    canvas_clicked = Signal(int, int)
    canvas_double_clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建场景
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # 视图设置
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState)
        self.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing
        )

        # 启用变换锚点为鼠标位置
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # 组件状态
        self.background_image: Optional[np.ndarray] = None
        self.layer_manager: Optional[LayerManager] = None
        self.background_item: Optional[QGraphicsPixmapItem] = None

        # 素材图形项
        self.material_items: Dict[MaterialInstance, MaterialGraphicsItem] = {}

        # 泊松融合缓存
        self.poisson_cache = PoissonBlendCache()

        # 合成缓存
        self.composite_cache: Optional[RenderCache] = None
        self.composite_dirty = True

        # 性能统计
        self.render_time = 0.0
        self.last_render_time = 0.0

        # 优化标志
        self.enable_poisson_cache = True
        self.enable_incremental_render = True
        self.max_render_items = 50  # 同时渲染的最大项目数

        # 更新定时器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update)

        # 交互状态
        self._right_mouse_pressed = False
        self._last_mouse_pos = QPointF()

    def set_background_image(self, image: np.ndarray):
        """设置背景图像"""
        self.background_image = image.copy()

        # 更新背景项
        if self.background_item:
            self.scene.removeItem(self.background_item)

        qimage = numpy_to_qimage(image)
        if qimage:
            pixmap = QPixmap.fromImage(qimage)
            self.background_item = self.scene.addPixmap(pixmap)
            self.background_item.setZValue(-1000)  # 置于最底层

        self.composite_dirty = True
        self._schedule_update()

    def set_layer_manager(self, layer_manager: LayerManager):
        """设置图层管理器"""
        self.layer_manager = layer_manager
        self._update_material_items()

    def _update_material_items(self):
        """更新素材图形项"""
        if not self.layer_manager:
            return

        # 获取当前所有实例
        current_instances = set(self.layer_manager.get_all_instances())
        existing_instances = set(self.material_items.keys())

        # 删除不存在的项
        for instance in existing_instances - current_instances:
            item = self.material_items[instance]
            self.scene.removeItem(item)
            del self.material_items[instance]

        # 添加新项
        for instance in current_instances - existing_instances:
            self._add_material_item(instance)

        self.composite_dirty = True
        self._schedule_update()

    def _add_material_item(self, instance: MaterialInstance):
        """添加素材图形项"""
        item = MaterialGraphicsItem(instance)
        item.set_graphics_view(self)  # 设置视图引用
        self.material_items[instance] = item

        # 添加到场景
        self.scene.addItem(item)
        item.setZValue(instance.layer_id)

        # 根据混合模式设置特殊处理
        if instance.blend_mode in ["poisson_normal", "poisson_mixed"]:
            # 泊松融合项显示轮廓以便交互，但在背景层统一渲染
            item.setVisible(True)  # 显示轮廓用于交互
            item.setOpacity(1.0)
        else:
            # 普通混合模式项正常显示
            item.setVisible(True)
            item.setOpacity(1.0)

    def _schedule_update(self):
        """安排更新"""
        if not self.update_timer.isActive():
            self.update_timer.start(16)  # 约60FPS

    def _delayed_update(self):
        """延迟更新"""
        if self.composite_dirty:
            self._render_composite()
            self.composite_dirty = False

    def _render_composite(self):
        """渲染合成图像"""
        try:
            start_time = time.time()

            if self.background_image is None or not self.layer_manager:
                return

            # 获取所有可见实例
            try:
                all_instances = [
                    instance
                    for instance in self.layer_manager.get_all_instances()
                    if instance.visible
                ]
            except Exception as e:
                print(f"获取实例失败: {e}")
                all_instances = []

            if not all_instances:
                # 没有实例，只显示背景
                try:
                    qimage = numpy_to_qimage(self.background_image)
                    if qimage and not qimage.isNull() and self.background_item:
                        pixmap = QPixmap.fromImage(qimage)
                        if not pixmap.isNull():
                            self.background_item.setPixmap(pixmap)
                except Exception as e:
                    print(f"更新背景显示失败: {e}")
                return

            # 分离普通混合和泊松融合实例
            try:
                normal_instances = [
                    instance
                    for instance in all_instances
                    if instance.blend_mode == "normal"
                ]
                poisson_instances = [
                    instance
                    for instance in all_instances
                    if instance.blend_mode in ["poisson_normal", "poisson_mixed"]
                ]
            except Exception as e:
                print(f"分离实例失败: {e}")
                normal_instances = []
                poisson_instances = []

            # 如果有泊松融合实例，需要重新合成整个背景
            if poisson_instances:
                try:
                    # 从背景开始合成
                    composite = self.background_image.copy()

                    # 处理泊松融合实例（普通实例由QGraphicsView自己渲染）
                    for instance in poisson_instances:
                        try:
                            composite = self._render_poisson_instance(
                                composite, instance
                            )
                        except Exception as e:
                            print(f"泊松融合失败 {instance.material_name}: {e}")
                            # 回退到普通混合
                            try:
                                composite = self._render_normal_instance(
                                    composite, instance
                                )
                            except:
                                print(f"回退普通混合也失败: {instance.material_name}")
                                continue

                    # 更新背景显示
                    try:
                        qimage = numpy_to_qimage(composite)
                        if qimage and not qimage.isNull() and self.background_item:
                            pixmap = QPixmap.fromImage(qimage)
                            if not pixmap.isNull():
                                self.background_item.setPixmap(pixmap)
                            else:
                                print("警告：合成图像QPixmap创建失败")
                        else:
                            print("警告：合成图像QImage创建失败或背景项为空")
                    except Exception as e:
                        print(f"更新背景显示失败: {e}")

                except Exception as e:
                    print(f"合成处理失败: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                # 没有泊松融合实例，只显示纯背景（普通实例由QGraphicsView处理）
                try:
                    qimage = numpy_to_qimage(self.background_image)
                    if qimage and not qimage.isNull() and self.background_item:
                        pixmap = QPixmap.fromImage(qimage)
                        if not pixmap.isNull():
                            self.background_item.setPixmap(pixmap)
                except Exception as e:
                    print(f"更新纯背景失败: {e}")

            self.render_time = time.time() - start_time
            self.last_render_time = time.time()

        except Exception as e:
            print(f"渲染合成图像失败: {e}")
            import traceback

            traceback.print_exc()

    def _render_normal_instance(
        self, canvas: np.ndarray, instance: MaterialInstance
    ) -> np.ndarray:
        """渲染普通混合模式实例"""
        # 复用原有的绘制逻辑
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

            if (
                src_x2 <= src_x1
                or src_y2 <= src_y1
                or dst_x2 <= dst_x1
                or dst_y2 <= dst_y1
            ):
                return canvas

            # 提取有效区域
            src_image = transformed_image[src_y1:src_y2, src_x1:src_x2]
            src_mask = transformed_mask[src_y1:src_y2, src_x1:src_x2]

            # 普通混合
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
        # 生成缓存键
        cache_key = self._generate_poisson_cache_key(canvas, instance)

        # 尝试从缓存获取
        if self.enable_poisson_cache:
            cached_result = self.poisson_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # 执行泊松融合
        try:
            # 获取不包含颜色叠加的变换图像进行泊松融合
            if hasattr(instance, "_get_transformed_image_without_color"):
                transformed_image = instance._get_transformed_image_without_color()
            else:
                # 临时保存颜色叠加设置
                original_color = instance.color_overlay
                original_opacity = instance.overlay_opacity

                # 临时清除颜色叠加
                instance.color_overlay = None
                instance.overlay_opacity = 0.0

                transformed_image = instance.get_transformed_image()

                # 恢复颜色叠加设置
                instance.color_overlay = original_color
                instance.overlay_opacity = original_opacity

            transformed_mask = instance.get_transformed_mask()

            if transformed_image is None or transformed_mask is None:
                print(f"泊松融合失败：图像或mask为空 - {instance.material_name}")
                return canvas

            # 检查图像和mask尺寸
            img_h, img_w = transformed_image.shape[:2]
            if img_h <= 0 or img_w <= 0:
                print(
                    f"泊松融合失败：图像尺寸无效 {img_w}x{img_h} - {instance.material_name}"
                )
                return canvas

            # 检查mask尺寸
            if transformed_mask is not None:
                mask_h, mask_w = transformed_mask.shape[:2]
                if mask_h != img_h or mask_w != img_w:
                    print(
                        f"泊松融合失败：图像({img_w}x{img_h})与mask({mask_w}x{mask_h})尺寸不匹配 - {instance.material_name}"
                    )
                    # 调整mask尺寸
                    try:
                        transformed_mask = cv2.resize(transformed_mask, (img_w, img_h))
                    except Exception as e:
                        print(f"调整mask尺寸失败: {e}")
                        return self._render_normal_instance(canvas, instance)

            # 确保mask是单通道
            if len(transformed_mask.shape) == 3:
                mask = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY)
            else:
                mask = transformed_mask

            # 检查mask有效性
            if np.sum(mask > 10) == 0:
                print(f"泊松融合失败：mask为空 - {instance.material_name}")
                return canvas

            # 计算mask的有效区域和中心偏移（与边界矩形计算一致）
            coords = np.column_stack(np.where(mask > 10))
            if len(coords) == 0:
                print(f"泊松融合失败：mask无有效区域 - {instance.material_name}")
                return canvas

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # 计算图像中心和mask中心
            img_center_x, img_center_y = img_w // 2, img_h // 2
            mask_center_x = (x_min + x_max) // 2
            mask_center_y = (y_min + y_max) // 2

            # mask中心相对于图像中心的偏移
            mask_offset_x = mask_center_x - img_center_x
            mask_offset_y = mask_center_y - img_center_y

            # 计算泊松融合的实际中心点
            # 因为cv2.seamlessClone使用图像中心作为参考，我们需要调整中心点来补偿mask的偏移
            actual_center_x = int(instance.x + mask_offset_x)
            actual_center_y = int(instance.y + mask_offset_y)

            # 检查画布尺寸和调整后的中心点
            canvas_h, canvas_w = canvas.shape[:2]
            if (
                actual_center_x < 0
                or actual_center_x >= canvas_w
                or actual_center_y < 0
                or actual_center_y >= canvas_h
            ):
                print(
                    f"泊松融合失败：调整后中心点({actual_center_x}, {actual_center_y})超出画布范围({canvas_w}x{canvas_h}) - {instance.material_name}"
                )
                return canvas

            # 执行泊松融合
            clone_type = (
                cv2.NORMAL_CLONE
                if instance.blend_mode == "poisson_normal"
                else cv2.MIXED_CLONE
            )

            print(f"泊松融合参数:")
            print(f"  实例中心: ({instance.x}, {instance.y})")
            print(
                f"  图像尺寸: {img_w}x{img_h}, 图像中心: ({img_center_x}, {img_center_y})"
            )
            print(f"  mask范围: x[{x_min}, {x_max}], y[{y_min}, {y_max}]")
            print(f"  mask中心: ({mask_center_x}, {mask_center_y})")
            print(f"  mask偏移: ({mask_offset_x}, {mask_offset_y})")
            print(f"  调整后中心: ({actual_center_x}, {actual_center_y})")

            mask_center = (actual_center_x, actual_center_y)
            result = cv2.seamlessClone(
                transformed_image, canvas, mask, mask_center, clone_type
            )

            # 验证结果
            if result is None or result.shape != canvas.shape:
                print(f"泊松融合结果无效 - {instance.material_name}")
                return self._render_normal_instance(canvas, instance)

            # 泊松融合后应用颜色叠加到有效区域
            if instance.color_overlay and instance.overlay_opacity > 0:
                result = self._apply_color_overlay_to_poisson_result(
                    result, instance, transformed_mask, actual_center_x, actual_center_y
                )

            # 缓存结果
            if self.enable_poisson_cache:
                self.poisson_cache.set(cache_key, result)

            return result

        except Exception as e:
            print(f"泊松融合失败 {instance.material_name}: {e}")
            # 回退到普通混合
            return self._render_normal_instance(canvas, instance)

    def _apply_color_overlay_to_poisson_result(
        self,
        canvas: np.ndarray,
        instance: MaterialInstance,
        transformed_mask: np.ndarray,
        actual_center_x: int,
        actual_center_y: int,
    ) -> np.ndarray:
        """在泊松融合结果上应用颜色叠加，只对有效区域生效"""
        try:
            if not instance.color_overlay or instance.overlay_opacity <= 0:
                return canvas

            # 获取变换后的图像尺寸（与mask尺寸相同）
            mask_h, mask_w = transformed_mask.shape[:2]
            canvas_h, canvas_w = canvas.shape[:2]

            # 获取不包含颜色叠加的变换图像来计算偏移
            if hasattr(instance, "_get_transformed_image_without_color"):
                transformed_image = instance._get_transformed_image_without_color()
            else:
                # 临时保存颜色叠加设置
                original_color = instance.color_overlay
                original_opacity = instance.overlay_opacity

                # 临时清除颜色叠加
                instance.color_overlay = None
                instance.overlay_opacity = 0.0

                transformed_image = instance.get_transformed_image()

                # 恢复颜色叠加设置
                instance.color_overlay = original_color
                instance.overlay_opacity = original_opacity

            img_h, img_w = transformed_image.shape[:2]
            img_center_x, img_center_y = img_w // 2, img_h // 2

            # 计算mask的有效区域边界
            if len(transformed_mask.shape) == 3:
                gray_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY)
            else:
                gray_mask = transformed_mask

            # 找到非零区域（与_update_bounding_rect逻辑一致）
            coords = np.column_stack(np.where(gray_mask > 10))
            if len(coords) == 0:
                print(f"  颜色叠加失败：mask为空")
                return canvas

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # 计算mask边界相对于图像中心的偏移（与_update_bounding_rect逻辑一致）
            mask_center_x = (x_min + x_max) // 2
            mask_center_y = (y_min + y_max) // 2

            # mask中心相对于图像中心的偏移
            mask_offset_x = mask_center_x - img_center_x
            mask_offset_y = mask_center_y - img_center_y

            # 由于actual_center_x/y已经包含了mask偏移（在调用时传入），
            # 我们需要计算mask有效区域在画布上的实际位置
            # actual_center_x = instance.x + mask_offset_x，所以原始图像中心在画布上是：
            image_center_x = actual_center_x - mask_offset_x
            image_center_y = actual_center_y - mask_offset_y

            # 计算mask有效区域的边界
            mask_width = x_max - x_min + 1
            mask_height = y_max - y_min + 1

            # mask有效区域在画布上的位置（基于actual_center作为mask的实际中心）
            mask_left = actual_center_x - mask_width // 2
            mask_top = actual_center_y - mask_height // 2
            mask_right = mask_left + mask_width
            mask_bottom = mask_top + mask_height

            print(f"颜色叠加区域计算（修复后）:")
            print(f"  实例中心: ({instance.x}, {instance.y})")
            print(f"  实际绘制中心: ({actual_center_x}, {actual_center_y})")
            print(
                f"  图像尺寸: {img_w}x{img_h}, 图像中心像素: ({img_center_x}, {img_center_y})"
            )
            print(f"  mask范围: x[{x_min}, {x_max}], y[{y_min}, {y_max}]")
            print(f"  mask中心像素: ({mask_center_x}, {mask_center_y})")
            print(f"  mask偏移: ({mask_offset_x}, {mask_offset_y})")
            print(f"  图像中心在画布: ({image_center_x}, {image_center_y})")
            print(
                f"  mask有效区域: ({mask_left}, {mask_top}) 到 ({mask_right}, {mask_bottom})"
            )

            # 裁剪到画布范围内
            canvas_x1 = max(0, mask_left)
            canvas_y1 = max(0, mask_top)
            canvas_x2 = min(canvas_w, mask_right)
            canvas_y2 = min(canvas_h, mask_bottom)

            # 计算在mask中对应的区域（考虑偏移）
            mask_x1 = x_min + max(0, canvas_x1 - mask_left)
            mask_y1 = y_min + max(0, canvas_y1 - mask_top)
            mask_x2 = mask_x1 + (canvas_x2 - canvas_x1)
            mask_y2 = mask_y1 + (canvas_y2 - canvas_y1)

            print(
                f"  裁剪后画布区域: ({canvas_x1}, {canvas_y1}) 到 ({canvas_x2}, {canvas_y2})"
            )
            print(f"  对应mask区域: ({mask_x1}, {mask_y1}) 到 ({mask_x2}, {mask_y2})")

            # 检查区域有效性
            if canvas_x1 >= canvas_x2 or canvas_y1 >= canvas_y2:
                print(f"  颜色叠加区域无效，跳过")
                return canvas

            if mask_x1 >= mask_w or mask_y1 >= mask_h or mask_x2 <= 0 or mask_y2 <= 0:
                print(f"  mask区域无效，跳过")
                return canvas

            # 确保mask区域在有效范围内
            mask_x1 = max(0, min(mask_w, mask_x1))
            mask_y1 = max(0, min(mask_h, mask_y1))
            mask_x2 = max(0, min(mask_w, mask_x2))
            mask_y2 = max(0, min(mask_h, mask_y2))

            # 获取画布和mask的对应区域
            canvas_roi = canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2]
            mask_roi = transformed_mask[mask_y1:mask_y2, mask_x1:mask_x2]

            print(f"  实际处理区域: 画布{canvas_roi.shape}, mask{mask_roi.shape}")

            # 确保区域尺寸匹配
            if canvas_roi.shape[:2] != mask_roi.shape[:2]:
                print(
                    f"  区域尺寸不匹配: 画布{canvas_roi.shape[:2]} vs mask{mask_roi.shape[:2]}"
                )
                # 调整区域尺寸以匹配
                min_h = min(canvas_roi.shape[0], mask_roi.shape[0])
                min_w = min(canvas_roi.shape[1], mask_roi.shape[1])
                if min_h <= 0 or min_w <= 0:
                    print(f"  调整后区域无效，跳过")
                    return canvas
                canvas_roi = canvas_roi[:min_h, :min_w]
                mask_roi = mask_roi[:min_h, :min_w]
                canvas_x2 = canvas_x1 + min_w
                canvas_y2 = canvas_y1 + min_h

            # 确保mask是单通道
            if len(mask_roi.shape) == 3:
                mask_roi = cv2.cvtColor(mask_roi, cv2.COLOR_BGR2GRAY)

            # 创建颜色叠加
            overlay_color = (
                instance.color_overlay[2],
                instance.color_overlay[1],
                instance.color_overlay[0],
            )  # RGB转BGR
            color_layer = np.full_like(canvas_roi, overlay_color, dtype=np.uint8)

            # 创建alpha mask，只在有效区域应用颜色叠加
            alpha_mask = (
                mask_roi.astype(np.float32) / 255.0
            ) * instance.overlay_opacity

            # 扩展到3通道
            if len(alpha_mask.shape) == 2:
                alpha_mask = np.stack([alpha_mask] * 3, axis=2)

            # 应用颜色叠加，只在mask有效区域
            blended_roi = (
                canvas_roi.astype(np.float32) * (1 - alpha_mask)
                + color_layer.astype(np.float32) * alpha_mask
            ).astype(np.uint8)

            # 更新画布
            canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = blended_roi

            print(f"  颜色叠加应用完成")

            return canvas

        except Exception as e:
            print(f"泊松融合颜色叠加失败: {e}")
            import traceback

            traceback.print_exc()
            return canvas

    def _generate_poisson_cache_key(
        self, canvas: np.ndarray, instance: MaterialInstance
    ) -> str:
        """生成泊松融合缓存键"""
        # 使用实例关键属性和画布区域哈希
        canvas_hash = hash(canvas.data.tobytes())
        instance_hash = hash(
            (
                id(instance),
                instance.x,
                instance.y,
                instance.scale,
                instance.rotation,
                instance.blend_mode,
                str(instance.color_overlay),
                instance.overlay_opacity,
            )
        )

        return f"poisson_{canvas_hash}_{instance_hash}"

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        try:
            # 获取缩放因子
            zoom_in_factor = 1.25
            zoom_out_factor = 1 / zoom_in_factor

            # 保存当前鼠标位置
            old_pos = self.mapToScene(event.position().toPoint())

            # 获取当前变换
            current_transform = self.transform()

            # 设置缩放 - 只缩放视图，不影响背景图像
            if event.angleDelta().y() > 0:
                self.scale(zoom_in_factor, zoom_in_factor)
            else:
                self.scale(zoom_out_factor, zoom_out_factor)

            # 获取缩放后的鼠标位置
            new_pos = self.mapToScene(event.position().toPoint())

            # 调整视图位置，使鼠标位置保持不变
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())

            # 确保背景项不受变换影响
            if self.background_item:
                # 背景项应该保持固定的变换
                self.background_item.setTransform(QTransform())  # 重置变换
                self.background_item.setZValue(-1000)  # 确保在最底层

        except Exception as e:
            print(f"缩放事件处理失败: {e}")
            import traceback

            traceback.print_exc()

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.RightButton:
            # 右键拖动视图
            self._right_mouse_pressed = True
            self._last_mouse_pos = event.position()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # 左键点击
            scene_pos = self.mapToScene(event.position().toPoint())

            # 检查是否点击了素材项
            item = self.itemAt(event.position().toPoint())

            # 调试输出
            print(f"点击位置: {event.position().toPoint()}, 场景位置: {scene_pos}")
            print(f"点击的项目: {item}, 类型: {type(item)}")

            if item and isinstance(item, MaterialGraphicsItem):
                # 选择素材项
                print(f"选择素材实例: {item.instance.material_name}")
                self.scene.clearSelection()
                item.setSelected(True)
                self.instance_selected.emit(item.instance)
            else:
                # 检查是否点击了其他类型的图形项（如背景）
                if item and hasattr(item, "instance"):
                    print(f"选择其他类型的实例: {type(item)}")
                    self.scene.clearSelection()
                    item.setSelected(True)
                    self.instance_selected.emit(item.instance)
                else:
                    # 点击空白区域
                    print("点击空白区域")
                    self.scene.clearSelection()
                    self.canvas_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if self._right_mouse_pressed:
            # 右键拖动视图
            delta = event.position() - self._last_mouse_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            self._last_mouse_pos = event.position()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.RightButton:
            self._right_mouse_pressed = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """鼠标双击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())

            # 检查是否双击了素材项
            item = self.itemAt(event.position().toPoint())
            if not item or not hasattr(item, "instance"):
                # 双击空白区域
                self.canvas_double_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))

        super().mouseDoubleClickEvent(event)

    def get_render_stats(self) -> Dict[str, Any]:
        """获取渲染统计信息"""
        return {
            "render_time": self.render_time,
            "last_render_time": self.last_render_time,
            "cache_size": len(self.poisson_cache._cache),
            "material_items": len(self.material_items),
            "poisson_cache_enabled": self.enable_poisson_cache,
            "incremental_render_enabled": self.enable_incremental_render,
        }

    def update_instance_display(self, instance: MaterialInstance):
        """更新指定实例的显示"""
        if instance in self.material_items:
            item = self.material_items[instance]
            item.update_from_instance()
            # 标记需要重新合成
            self.composite_dirty = True
            self._schedule_update()


class HighPerformanceCanvasWidget(QWidget):
    """高性能画布小部件"""

    # 信号定义
    instance_selected = Signal(object)
    instance_moved = Signal(object)
    canvas_clicked = Signal(int, int)
    canvas_double_clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建图形视图
        self.graphics_view = HighPerformanceGraphicsView()
        layout.addWidget(self.graphics_view)

        # 连接信号
        self.graphics_view.instance_selected.connect(self.instance_selected)
        self.graphics_view.instance_moved.connect(self.instance_moved)
        self.graphics_view.canvas_clicked.connect(self.canvas_clicked)
        self.graphics_view.canvas_double_clicked.connect(self.canvas_double_clicked)

        # 兼容性属性
        self.incremental_update_enabled = True

    @property
    def background_image(self):
        """获取背景图像（兼容性属性）"""
        return self.graphics_view.background_image

    @property
    def selected_instance(self):
        """获取当前选择的实例（兼容性属性）"""
        selected_items = self.graphics_view.scene.selectedItems()
        if selected_items and hasattr(selected_items[0], "instance"):
            return selected_items[0].instance
        return None

    @selected_instance.setter
    def selected_instance(self, value):
        """设置当前选择的实例（兼容性属性）"""
        # 清除所有选择
        self.graphics_view.scene.clearSelection()

        # 如果设置为None，直接返回
        if value is None:
            return

        # 查找对应的图形项并选择
        for instance, item in self.graphics_view.material_items.items():
            if instance == value:
                item.setSelected(True)
                break

    def set_background_image(self, image: np.ndarray):
        """设置背景图像"""
        self.graphics_view.set_background_image(image)

    def set_layer_manager(self, layer_manager: LayerManager):
        """设置图层管理器"""
        self.graphics_view.set_layer_manager(layer_manager)

    def update_canvas(self, force_full_update: bool = False):
        """更新画布"""
        # 先更新素材图形项
        self.graphics_view._update_material_items()

        # 然后标记合成需要更新
        self.graphics_view.composite_dirty = True
        self.graphics_view._schedule_update()

    def zoom_to_fit(self):
        """缩放到适合窗口"""
        self.graphics_view.fitInView(
            self.graphics_view.scene.itemsBoundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def zoom_to_actual_size(self):
        """缩放到实际大小"""
        self.graphics_view.resetTransform()

    def get_render_stats(self) -> Dict[str, Any]:
        """获取渲染统计信息"""
        return self.graphics_view.get_render_stats()

    def update_instance_display(self, instance: MaterialInstance):
        """更新指定实例的显示"""
        self.graphics_view.update_instance_display(instance)

    def _draw_instance_on_canvas(
        self, canvas: np.ndarray, instance: MaterialInstance
    ) -> np.ndarray:
        """在画布上绘制实例（兼容性方法）"""
        try:
            if instance.blend_mode in ["poisson_normal", "poisson_mixed"]:
                # 泊松融合 - 使用修复后的逻辑
                # 获取不包含颜色叠加的变换图像进行泊松融合
                if hasattr(instance, "_get_transformed_image_without_color"):
                    transformed_image = instance._get_transformed_image_without_color()
                else:
                    # 临时保存颜色叠加设置
                    original_color = instance.color_overlay
                    original_opacity = instance.overlay_opacity

                    # 临时清除颜色叠加
                    instance.color_overlay = None
                    instance.overlay_opacity = 0.0

                    transformed_image = instance.get_transformed_image()

                    # 恢复颜色叠加设置
                    instance.color_overlay = original_color
                    instance.overlay_opacity = original_opacity

                transformed_mask = instance.get_transformed_mask()

                if transformed_image is None or transformed_mask is None:
                    return canvas

                # 确保mask是单通道
                if len(transformed_mask.shape) == 3:
                    mask = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY)
                else:
                    mask = transformed_mask

                # 计算mask的有效区域和中心偏移（与主渲染逻辑一致）
                coords = np.column_stack(np.where(mask > 10))
                if len(coords) == 0:
                    return canvas

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                # 计算图像中心和mask中心
                img_h, img_w = transformed_image.shape[:2]
                img_center_x, img_center_y = img_w // 2, img_h // 2
                mask_center_x = (x_min + x_max) // 2
                mask_center_y = (y_min + y_max) // 2

                # mask中心相对于图像中心的偏移
                mask_offset_x = mask_center_x - img_center_x
                mask_offset_y = mask_center_y - img_center_y

                # 计算泊松融合的实际中心点
                actual_center_x = int(instance.x + mask_offset_x)
                actual_center_y = int(instance.y + mask_offset_y)

                # 执行泊松融合
                clone_type = (
                    cv2.NORMAL_CLONE
                    if instance.blend_mode == "poisson_normal"
                    else cv2.MIXED_CLONE
                )

                mask_center = (actual_center_x, actual_center_y)
                result = cv2.seamlessClone(
                    transformed_image, canvas, mask, mask_center, clone_type
                )

                return result
            else:
                # 普通混合
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

                return canvas

        except Exception as e:
            print(f"绘制实例失败: {e}")
            return canvas
