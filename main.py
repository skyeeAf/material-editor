import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from harmonize import (
    get_available_harmonize_backends,
    harmonize_region,
    repair_seam,
    set_harmonize_backend,
)
from patchmatch_inpaint import (
    InpaintBackend,
    _get_backend_name,
    get_available_backends,
    get_backend,
    patchmatch_inpaint,
    set_backend,
)
from ui.dialogs import RandomGenerateDialog


def cv_imread_rgba(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"无法读取图像: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img  # BGRA


def cv_to_qimage_bgra(bgra: np.ndarray) -> QImage:
    h, w = bgra.shape[:2]
    return (
        QImage(bgra.data, w, h, bgra.strides[0], QImage.Format.Format_RGBA8888)
        .rgbSwapped()
        .copy()
    )


def tint_bgra(
    bgra: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float
) -> np.ndarray:
    if alpha <= 0:
        return bgra
    out = bgra.copy()
    overlay = np.zeros_like(out[:, :, :3], dtype=np.float32)
    overlay[:, :] = np.array(color_bgr, dtype=np.float32)
    src = out[:, :, :3].astype(np.float32)
    src = (1.0 - alpha) * src + alpha * overlay
    out[:, :, :3] = np.clip(src, 0, 255).astype(np.uint8)
    return out


def dominant_color_bgr(img_bgr: np.ndarray, k: int = 4) -> Tuple[int, int, int]:
    small = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    z = small.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # 为避免静态类型检查器报错，提供初始 bestLabels
    best_labels = np.empty((z.shape[0], 1), dtype=np.int32)
    _, labels, centers = cv2.kmeans(
        z, k, best_labels, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten())
    dom = centers[np.argmax(counts)]
    return int(dom[0]), int(dom[1]), int(dom[2])


def dominant_colors_bgr(img_bgr: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
    """返回按像素数量排序的前 k 个主色（BGR）。"""
    small = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    z = small.reshape((-1, 3)).astype(np.float32)
    if z.shape[0] == 0:
        return []
    k = max(1, min(k, z.shape[0]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    best_labels = np.empty((z.shape[0], 1), dtype=np.int32)
    _, labels, centers = cv2.kmeans(
        z, k, best_labels, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(counts)[::-1]
    colors: List[Tuple[int, int, int]] = []
    for idx in order:
        c = centers[idx]
        colors.append((int(c[0]), int(c[1]), int(c[2])))
    return colors


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MASK_SUFFIXES = ["_mask", "-mask", ".mask"]


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in IMAGE_EXTS


def scan_images_recursively(root_dir: str) -> List[str]:
    results: List[str] = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            p = os.path.join(base, fn)
            if is_image_file(p):
                results.append(p)
    results.sort()
    return results


def material_pairs_from_dir(root_dir: str) -> List[Tuple[str, Optional[str]]]:
    files = scan_images_recursively(root_dir)
    by_key: Dict[str, Dict[str, Optional[str]]] = {}
    for p in files:
        name = os.path.basename(p)
        stem, _ = os.path.splitext(name)
        lower = stem.lower()
        is_mask = False
        base_stem = stem
        for suf in MASK_SUFFIXES:
            if lower.endswith(suf):
                is_mask = True
                base_stem = stem[: -len(suf)]
                break
        key = os.path.join(os.path.dirname(p), base_stem).lower()
        ent = by_key.setdefault(key, {"img": None, "mask": None})
        if is_mask:
            ent["mask"] = p
        else:
            if ent["img"] is None:
                ent["img"] = p
    pairs: List[Tuple[str, Optional[str]]] = []
    for ent in by_key.values():
        if ent["img"] is not None:
            pairs.append((ent["img"], ent["mask"]))
    return pairs


def apply_mask_to_bgra(img_bgra: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    h, w = img_bgra.shape[:2]
    if mask_gray.ndim == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img_bgra.copy()
    out[:, :, 3] = mask_gray
    return out


def crop_to_alpha_bbox(img_bgra: np.ndarray) -> np.ndarray:
    """根据 alpha 通道裁剪到最小外接矩形，使物体框线贴合掩码区域。"""
    if img_bgra.shape[2] < 4:
        return img_bgra
    alpha = img_bgra[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        return img_bgra
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return img_bgra[y0:y1, x0:x1]


class BlendMode:
    PASTE = 0
    POISSON_NORMAL = 1
    POISSON_MIXED = 2
    KONTEXT_BLEND = 3
    KONTEXT_HARMONIZE = 4


class _HqSignals(QObject):
    finished = Signal(int, object)  # (serial, canvas_bgr)


class _InpaintSignals(QObject):
    """内容识别填充异步结果信号。"""

    finished = Signal(object)  # result_bgr: np.ndarray
    error = Signal(str)  # 错误信息


class GLGraphicsView(QGraphicsView):
    """自定义 QGraphicsView，支持缩放、背景取色和套索选区。"""

    zoomChanged = Signal(float)
    requestPickBackgroundColor = Signal(int, int)
    mousePressed = Signal()
    mouseReleased = Signal()
    lassoCompleted = Signal(list)

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        try:
            self.setViewport(QOpenGLWidget())
        except Exception:
            pass
        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._zoom = 1.0
        self._picking_bg_color = False
        # 套索模式
        self._lasso_mode = False
        self._lasso_drawing = False
        self._lasso_points: List[QPointF] = []
        self._lasso_path_item: Optional[QGraphicsPathItem] = None

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else 1 / 1.15
        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        self._zoom *= factor
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        self.zoomChanged.emit(self._zoom)

    def enable_pick_background_color(self, enable: bool):
        self._picking_bg_color = enable
        self.setCursor(
            Qt.CursorShape.CrossCursor if enable else Qt.CursorShape.ArrowCursor
        )

    def enable_lasso_mode(self, enable: bool):
        """进入或退出套索选区模式。"""
        self._lasso_mode = enable
        self._lasso_drawing = False
        if enable:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.clear_lasso_path()

    def clear_lasso_path(self):
        """清除套索路径可视化。"""
        if self._lasso_path_item is not None:
            self.scene().removeItem(self._lasso_path_item)
            self._lasso_path_item = None
        self._lasso_points.clear()
        self._lasso_drawing = False

    def _update_lasso_visual(self):
        """根据 _lasso_points 更新套索路径可视化。"""
        if not self._lasso_points:
            return
        path = QPainterPath()
        path.moveTo(self._lasso_points[0])
        for pt in self._lasso_points[1:]:
            path.lineTo(pt)
        if self._lasso_path_item is None:
            pen = QPen(QColor(0, 255, 255, 200), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._lasso_path_item = QGraphicsPathItem(path)
            self._lasso_path_item.setPen(pen)
            self._lasso_path_item.setBrush(QColor(0, 255, 255, 30))
            self._lasso_path_item.setZValue(10000)
            self.scene().addItem(self._lasso_path_item)
        else:
            self._lasso_path_item.setPath(path)

    def mousePressEvent(self, event):
        if self._picking_bg_color and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.requestPickBackgroundColor.emit(int(scene_pos.x()), int(scene_pos.y()))
            return
        if self._lasso_mode and event.button() == Qt.MouseButton.LeftButton:
            self.clear_lasso_path()
            self._lasso_drawing = True
            scene_pos = self.mapToScene(event.position().toPoint())
            self._lasso_points.append(scene_pos)
            return
        self.mousePressed.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._lasso_mode and self._lasso_drawing:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._lasso_points.append(scene_pos)
            self._update_lasso_visual()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self._lasso_mode
            and self._lasso_drawing
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._lasso_drawing = False
            if len(self._lasso_points) < 10:
                self.clear_lasso_path()
                return
            # 闭合路径并更新可视化
            self._lasso_points.append(self._lasso_points[0])
            self._update_lasso_visual()
            # 发射完成信号（传递副本）
            self.lassoCompleted.emit(list(self._lasso_points))
            return
        super().mouseReleaseEvent(event)
        self.mouseReleased.emit()


class RotationHandleItem(QGraphicsItem):
    """选中素材四角处的弧形旋转箭头，拖拽以中心为基准旋转。"""

    HANDLE_SIZE = 28

    def __init__(self, parent_item: "MaterialItem", corner: int):
        super().__init__(parent_item)
        self._parent_material = parent_item
        self._corner = corner  # 0=左上, 1=右上, 2=右下, 3=左下
        self._dragging = False
        self._drag_start_angle: Optional[float] = None
        self._drag_start_rot: Optional[int] = None
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setZValue(100)

    def _center_scene(self) -> QPointF:
        b = self._parent_material.boundingRect()
        return self._parent_material.mapToScene(b.center())

    def _point_to_angle_deg(self, scene_pos: QPointF) -> float:
        c = self._center_scene()
        dx = scene_pos.x() - c.x()
        dy = scene_pos.y() - c.y()
        if dx == 0 and dy == 0:
            return 0.0
        return math.degrees(math.atan2(-dy, dx))

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.HANDLE_SIZE, self.HANDLE_SIZE)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(0, 0, self.HANDLE_SIZE, self.HANDLE_SIZE)
        return path

    def paint(self, painter: QPainter, option, widget) -> None:
        path = self._arrow_path()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(220, 60, 60), 2))
        painter.setBrush(QColor(220, 60, 60, 180))
        painter.drawPath(path)

    def _arrow_path(self) -> QPainterPath:
        s = self.HANDLE_SIZE
        r = s * 0.35
        cx, cy = s / 2, s / 2
        rot = self._corner * 90
        start_deg = 210 + rot
        path = QPainterPath()
        path.moveTo(cx + r * math.cos(math.radians(start_deg)), cy - r * math.sin(math.radians(start_deg)))
        path.arcTo(2, 2, s - 4, s - 4, start_deg, -240)
        tip_deg = start_deg - 240
        tip_x = cx + r * math.cos(math.radians(tip_deg))
        tip_y = cy - r * math.sin(math.radians(tip_deg))
        wing = 5
        path.moveTo(tip_x, tip_y)
        path.lineTo(tip_x - wing * math.cos(math.radians(tip_deg - 22)), tip_y + wing * math.sin(math.radians(tip_deg - 22)))
        path.lineTo(tip_x - wing * 0.5 * math.cos(math.radians(tip_deg)), tip_y + wing * 0.5 * math.sin(math.radians(tip_deg)))
        path.lineTo(tip_x - wing * math.cos(math.radians(tip_deg + 22)), tip_y + wing * math.sin(math.radians(tip_deg + 22)))
        path.closeSubpath()
        return path

    def _update_pos(self) -> None:
        b = self._parent_material.boundingRect()
        o = 4
        if self._corner == 0:
            self.setPos(b.left() - o - self.HANDLE_SIZE, b.top() - o - self.HANDLE_SIZE)
        elif self._corner == 1:
            self.setPos(b.right() + o, b.top() - o - self.HANDLE_SIZE)
        elif self._corner == 2:
            self.setPos(b.right() + o, b.bottom() + o)
        else:
            self.setPos(b.left() - o - self.HANDLE_SIZE, b.bottom() + o)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_angle = self._point_to_angle_deg(event.scenePos())
            self._drag_start_rot = self._parent_material.rotation_deg
            if self._parent_material.host:
                self._parent_material.host._on_item_interaction_started(self._parent_material)
                self._parent_material.host._disable_hq_overlay()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._drag_start_angle is not None and self._drag_start_rot is not None:
            cur = self._point_to_angle_deg(event.scenePos())
            delta = cur - self._drag_start_angle
            new_rot = (self._drag_start_rot + int(round(delta))) % 360
            self._parent_material.set_rotation_deg(new_rot)
            if self._parent_material.host:
                self._parent_material.host._sync_rotation_from_item(self._parent_material)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_start_angle = None
            self._drag_start_rot = None
            if self._parent_material.host:
                self._parent_material.host._on_item_interaction_finished(self._parent_material)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class MaterialItem(QGraphicsPixmapItem):
    def __init__(self, name: str, path: str, src_bgra: np.ndarray, host: "MainWindow"):
        super().__init__()
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.name = name
        self.path = path
        self.base_bgra = src_bgra  # original BGRA
        self.blend_mode = BlendMode.PASTE
        self.scale_ratio = 1.0
        self.rotation_deg = 0
        self.tint_color_bgr: Tuple[int, int, int] = (0, 0, 0)
        self.tint_alpha = 0.0  # 0~1
        self.mask_offset = 0  # >0 膨胀, <0 腐蚀
        self.strong_tint = False  # 强叠加模式
        self.feather_radius = 0  # 羽化半径 px (0~50)
        self.brightness = 0  # 亮度偏移 (-100~100)
        self.contrast = 100  # 对比度百分比 (50~200, 100=原始)
        self.hue_shift = 0  # 色相偏移 (-180~180)
        self.saturation = 100  # 饱和度百分比 (0~300, 100=原始)
        self.gaussian_blur_radius = 0  # 高斯模糊半径 px (0~50)
        self.seam_repair = False  # 接缝修复（LaMa/inpaint）
        self.harmonize = False  # 色调协调
        self.host = host
        self._rotation_handles = [
            RotationHandleItem(self, i) for i in range(4)
        ]
        self._update_pix()

    def _update_rotation_handles(self) -> None:
        for h in self._rotation_handles:
            h._update_pos()
            h.setVisible(self.isSelected())

    def _make_display_qpixmap(self) -> QPixmap:
        img = self._make_transformed_bgra_for_display()
        return QPixmap.fromImage(cv_to_qimage_bgra(img))

    def _make_transformed_bgra_for_display(self) -> np.ndarray:
        img = self.base_bgra.copy()
        # 掩码腐蚀/膨胀（基于 alpha 通道）
        if self.mask_offset != 0 and img.shape[2] >= 4:
            alpha = img[:, :, 3]
            k = min(20, max(1, abs(int(self.mask_offset))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)
            )
            if self.mask_offset > 0:
                alpha = cv2.dilate(alpha, kernel, iterations=1)
            else:
                alpha = cv2.erode(alpha, kernel, iterations=1)
            img[:, :, 3] = alpha
        # 羽化（对 alpha 通道做高斯模糊）
        if self.feather_radius > 0 and img.shape[2] >= 4:
            ksize = self.feather_radius * 2 + 1
            img[:, :, 3] = cv2.GaussianBlur(img[:, :, 3], (ksize, ksize), 0)
        # 颜色叠加（预览阶段也支持强叠加）
        eff_alpha = self.tint_alpha
        if self.strong_tint and eff_alpha > 0:
            eff_alpha = min(1.0, eff_alpha * 1.5)
        if eff_alpha > 0:
            img = tint_bgra(img, self.tint_color_bgr, eff_alpha)
        # 亮度 / 对比度（仅处理 BGR 通道）
        if self.brightness != 0 or self.contrast != 100:
            bgr = img[:, :, :3]
            bgr = cv2.convertScaleAbs(bgr, alpha=self.contrast / 100.0, beta=self.brightness)
            img[:, :, :3] = bgr
        # 色相偏移 / 饱和度（仅处理 BGR 通道）
        if self.hue_shift != 0 or self.saturation != 100:
            bgr = img[:, :, :3]
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation / 100, 0, 255)
            img[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # 高斯模糊（仅处理 BGR 通道）
        if self.gaussian_blur_radius > 0:
            ksize = self.gaussian_blur_radius * 2 + 1
            img[:, :, :3] = cv2.GaussianBlur(img[:, :, :3], (ksize, ksize), 0)
        if self.scale_ratio != 1.0:
            img = cv2.resize(
                img,
                (0, 0),
                fx=self.scale_ratio,
                fy=self.scale_ratio,
                interpolation=cv2.INTER_LINEAR,
            )
        if self.rotation_deg % 360 != 0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), self.rotation_deg, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))
            M[0, 2] += (nw / 2) - w / 2
            M[1, 2] += (nh / 2) - h / 2
            img = cv2.warpAffine(
                img,
                M,
                (nw, nh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
        return img

    def _update_pix(self):
        self.setPixmap(self._make_display_qpixmap())
        b = self.boundingRect()
        self.setTransformOriginPoint(b.center())
        self._update_rotation_handles()

    def set_blend_mode(self, mode: int):
        self.blend_mode = mode

    def set_scale_ratio(self, r: float):
        self.scale_ratio = max(0.05, min(8.0, r))
        self._update_pix()

    def set_rotation_deg(self, deg: int):
        self.rotation_deg = deg % 360
        self._update_pix()

    def set_tint(self, color: Tuple[int, int, int], alpha: float):
        self.tint_color_bgr = color
        self.tint_alpha = max(0.0, min(1.0, alpha))
        self._update_pix()

    def set_mask_offset(self, v: int):
        self.mask_offset = int(max(-20, min(20, v)))
        self._update_pix()

    def set_strong_tint(self, v: bool):
        self.strong_tint = bool(v)
        self._update_pix()

    def set_feather_radius(self, v: int):
        """设置羽化半径（像素）。"""
        self.feather_radius = int(max(0, min(50, v)))
        self._update_pix()

    def set_brightness(self, v: int):
        """设置亮度偏移。"""
        self.brightness = int(max(-100, min(100, v)))
        self._update_pix()

    def set_contrast(self, v: int):
        """设置对比度百分比。"""
        self.contrast = int(max(50, min(200, v)))
        self._update_pix()

    def set_hue_shift(self, v: int):
        """设置色相偏移。"""
        self.hue_shift = int(max(-180, min(180, v)))
        self._update_pix()

    def set_saturation(self, v: int):
        """设置饱和度百分比。"""
        self.saturation = int(max(0, min(300, v)))
        self._update_pix()

    def set_gaussian_blur_radius(self, v: int):
        """设置高斯模糊半径（像素）。"""
        self.gaussian_blur_radius = int(max(0, min(50, v)))
        self._update_pix()

    def to_composite_package(self) -> dict:
        """导出合成所需的完整数据包。"""
        disp = self._make_transformed_bgra_for_display()
        pos = self.scenePos()
        cx = int(pos.x() + self.boundingRect().width() / 2)
        cy = int(pos.y() + self.boundingRect().height() / 2)
        mask = disp[:, :, 3]
        return {
            "name": self.name,
            "path": self.path,
            "img_bgra": disp,
            "mask": mask,
            "center": (cx, cy),
            "mode": self.blend_mode,
            "z": float(self.zValue()),
            "tint_color_bgr": tuple(self.tint_color_bgr),
            "tint_alpha": float(self.tint_alpha),
            "strong_tint": bool(self.strong_tint),
            "seam_repair": bool(self.seam_repair),
            "harmonize": bool(self.harmonize),
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为状态字典（用于保存/撤销）。"""
        return {
            "name": self.name,
            "path": self.path,
            "pos": (float(self.scenePos().x()), float(self.scenePos().y())),
            "scale": float(self.scale_ratio),
            "rotation": int(self.rotation_deg),
            "tint_color_bgr": tuple(self.tint_color_bgr),
            "tint_alpha": float(self.tint_alpha),
            "mode": int(self.blend_mode),
            "z": float(self.zValue()),
            "mask_offset": int(self.mask_offset),
            "strong_tint": bool(self.strong_tint),
            "feather_radius": int(self.feather_radius),
            "brightness": int(self.brightness),
            "contrast": int(self.contrast),
            "hue_shift": int(self.hue_shift),
            "saturation": int(self.saturation),
            "gaussian_blur_radius": int(self.gaussian_blur_radius),
            "seam_repair": bool(self.seam_repair),
            "harmonize": bool(self.harmonize),
        }

    def itemChange(self, change, value):
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.host is not None
        ):
            self.host._on_item_interaction_started(self)
            self.host._disable_hq_overlay()
        elif (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
            and self.host is not None
        ):
            self.host._on_item_interaction_finished(self)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            visible = bool(value)
            for h in self._rotation_handles:
                h._update_pos()
                h.setVisible(visible)
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._update_rotation_handles()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if self.host is not None:
            self.host._on_item_interaction_started(self)
            self.host._disable_hq_overlay()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.host is not None:
            self.host._on_item_interaction_finished(self)


class ContentAwareFillDialog(QDialog):
    """内容识别填充参数对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("内容识别填充 (PatchMatch)")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        # 后端选择
        self.cmb_backend = QComboBox()
        backends = get_available_backends()
        current = get_backend()
        select_idx = 0
        for i, (backend_enum, display_name, available) in enumerate(backends):
            suffix = "" if available else " [不可用]"
            self.cmb_backend.addItem(display_name + suffix, userData=backend_enum)
            if not available:
                # 禁用不可用项
                self.cmb_backend.model().item(i).setEnabled(False)
            if backend_enum == current:
                select_idx = i
        self.cmb_backend.setCurrentIndex(select_idx)
        form.addRow("后端", self.cmb_backend)

        self.spn_patch = QSpinBox()
        self.spn_patch.setRange(3, 15)
        self.spn_patch.setValue(7)
        self.spn_patch.setSingleStep(2)
        form.addRow("patch 大小(px)", self.spn_patch)

        self.spn_expand = QSpinBox()
        self.spn_expand.setRange(0, 20)
        self.spn_expand.setValue(3)
        form.addRow("选区扩展(px)", self.spn_expand)

        self.spn_max_size = QSpinBox()
        self.spn_max_size.setRange(0, 4096)
        self.spn_max_size.setValue(0)
        self.spn_max_size.setSingleStep(128)
        self.spn_max_size.setSpecialValueText("不限制")
        self.spn_max_size.setToolTip(
            "ROI 长边上限（像素），0=不限制。降采样可大幅加速，但会损失细节。"
        )
        form.addRow("降采样上限(px)", self.spn_max_size)

        layout.addLayout(form)
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> Optional[Dict[str, Any]]:
        """显示对话框并返回参数字典，取消返回 None。"""
        if self.exec() == QDialog.DialogCode.Accepted:
            return {
                "patch_size": self.spn_patch.value(),
                "expand": self.spn_expand.value(),
                "backend": self.cmb_backend.currentData(),
                "max_size": self.spn_max_size.value(),
            }
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Material Editor v2 - 素材叠加编辑器")
        self.resize(1400, 900)

        self.scene = QGraphicsScene(self)
        self.view = GLGraphicsView(self.scene, self)
        self.view.setBackgroundBrush(QColor(30, 30, 30))
        self.view.requestPickBackgroundColor.connect(self._on_pick_bg_color)
        # 注：不在 view 级 mousePressed/mouseReleased 上触发 HQ 重渲染，
        # 因为平移/缩放不改变合成结果。素材拖动已通过
        # MaterialItem.itemChange → _on_item_interaction_finished 正确处理。

        self.bg_pix_item = QGraphicsPixmapItem()
        self.bg_pix_item.setZValue(-1000)
        self.scene.addItem(self.bg_pix_item)

        self.hq_overlay_item = QGraphicsPixmapItem()
        self.hq_overlay_item.setZValue(9000)  # 在素材之上，视觉覆盖
        self.hq_overlay_item.setVisible(False)
        # 禁止 overlay 接收鼠标事件，点击穿透到下方的素材
        self.hq_overlay_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.hq_overlay_item.setAcceptHoverEvents(False)
        self.scene.addItem(self.hq_overlay_item)

        self.current_bg_index = -1
        self.bg_list: List[str] = []
        self.bg_bgr: Optional[np.ndarray] = None  # for analysis/exports

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        center = QWidget(self)
        center_layout = QHBoxLayout(center)
        center_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_container = QWidget()
        llyt = QVBoxLayout(left_container)
        llyt.setContentsMargins(0, 0, 0, 0)
        llyt.addWidget(left_panel)
        right_container = QWidget()
        rlyt = QVBoxLayout(right_container)
        rlyt.setContentsMargins(0, 0, 0, 0)
        rlyt.addWidget(right_panel)

        splitter.addWidget(left_container)
        splitter.addWidget(self.view)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        center_layout.addWidget(splitter)
        self.setCentralWidget(center)

        self._build_toolbar()
        self._connect_signals()

        # state
        self.material_items: List[MaterialItem] = []
        self._suppress_ui = False

        # high-quality preview controls
        self.hq_enabled = False
        self.hq_timer = QTimer(self)
        self.hq_timer.setSingleShot(True)
        self.hq_timer.setInterval(350)
        self.hq_timer.timeout.connect(self._run_hq_preview_async)
        self.hq_serial = 0
        self._hq_future = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._hq_signals = _HqSignals(self)
        self._hq_signals.finished.connect(self._apply_hq_result)

        # content-aware fill async controls
        self._inpaint_signals = _InpaintSignals(self)
        self._inpaint_signals.finished.connect(self._on_inpaint_finished)
        self._inpaint_signals.error.connect(self._on_inpaint_error)
        self._inpaint_future = None

        # history (undo/redo)
        self.history: List[Dict[str, Any]] = []
        self.history_index: int = -1
        self.hist_timer = QTimer(self)
        self.hist_timer.setSingleShot(True)
        self.hist_timer.setInterval(400)
        self.hist_timer.timeout.connect(self._push_history)

        # 初始历史
        self._push_history()

    def _build_toolbar(self):
        tb = QToolBar("工具栏", self)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)

        self.act_load_bg = QAction("加载背景", self)
        self.act_load_material = QAction("加载素材", self)
        self.act_prev_bg = QAction("上一张", self)
        self.act_next_bg = QAction("下一张", self)
        self.act_clear_materials = QAction("清空素材", self)
        self.act_export = QAction("一键导出", self)
        self.act_hq_preview = QAction("高质量预览", self)
        self.act_hq_preview.setCheckable(True)
        self.act_random_generate = QAction("随机生成素材", self)
        self.act_lasso_fill = QAction("套索填充", self)
        self.act_lasso_fill.setCheckable(True)
        self.act_undo = QAction("撤销", self)
        self.act_undo.setShortcut("Ctrl+Z")
        self.act_redo = QAction("重做", self)
        self.act_redo.setShortcut("Ctrl+Y")

        tb.addAction(self.act_load_bg)
        tb.addAction(self.act_load_material)
        tb.addSeparator()
        tb.addAction(self.act_prev_bg)
        tb.addAction(self.act_next_bg)
        tb.addSeparator()
        tb.addAction(self.act_clear_materials)
        tb.addSeparator()
        tb.addAction(self.act_export)
        tb.addSeparator()
        tb.addAction(self.act_random_generate)
        tb.addSeparator()
        tb.addAction(self.act_hq_preview)
        tb.addSeparator()
        tb.addAction(self.act_lasso_fill)
        tb.addSeparator()
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("素材列表")
        self.list_materials = QListWidget()
        self.list_materials.setIconSize(QSize(64, 64))
        self.list_materials.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        vl.addWidget(lbl)
        vl.addWidget(self.list_materials, 1)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)

        grp_top = QGroupBox("已添加素材 + 选中素材属性")
        top_l = QVBoxLayout(grp_top)

        self.list_added = QListWidget()
        self.list_added.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        btn_row = QWidget()
        brl = QHBoxLayout(btn_row)
        brl.setContentsMargins(0, 0, 0, 0)
        self.btn_up = QPushButton("上移")
        self.btn_down = QPushButton("下移")
        self.btn_top = QPushButton("置顶")
        self.btn_bottom = QPushButton("置底")
        self.btn_delete_added = QPushButton("删除")
        for b in (
            self.btn_up,
            self.btn_down,
            self.btn_top,
            self.btn_bottom,
            self.btn_delete_added,
        ):
            brl.addWidget(b)

        top_l.addWidget(self.list_added)
        top_l.addWidget(btn_row)

        props = QGroupBox("属性")
        form = QFormLayout(props)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(
            [
                "直接粘贴",
                "泊松融合 Normal",
                "泊松融合 Mix",
                "AI 融合 (Kontext)",
                "AI 融合+协调 (Kontext)",
            ]
        )
        # 检测 Kontext 是否可用，不可用时灰显对应选项

        self.sld_rot = QSlider(Qt.Orientation.Horizontal)
        self.sld_rot.setRange(0, 359)
        self.spn_rot = QSpinBox()
        self.spn_rot.setRange(0, 359)
        self.sld_scale = QSlider(Qt.Orientation.Horizontal)
        self.sld_scale.setRange(10, 400)
        self.spn_scale = QSpinBox()
        self.spn_scale.setRange(10, 400)
        self.chk_tint = QCheckBox("启用颜色叠加")
        self.btn_tint_color = QPushButton("选择叠加颜色")
        self.sld_tint_alpha = QSlider(Qt.Orientation.Horizontal)
        self.sld_tint_alpha.setRange(0, 100)
        self.spn_tint_alpha = QSpinBox()
        self.spn_tint_alpha.setRange(0, 100)
        # 强叠加模式
        self.chk_tint_strong = QCheckBox("强叠加")
        # 掩码腐蚀/膨胀（负值腐蚀，正值膨胀）
        self.sld_mask_offset = QSlider(Qt.Orientation.Horizontal)
        self.sld_mask_offset.setRange(-20, 20)
        self.spn_mask_offset = QSpinBox()
        self.spn_mask_offset.setRange(-20, 20)
        # 羽化
        self.sld_feather = QSlider(Qt.Orientation.Horizontal)
        self.sld_feather.setRange(0, 50)
        self.spn_feather = QSpinBox()
        self.spn_feather.setRange(0, 50)

        rot_row = QWidget()
        rl = QHBoxLayout(rot_row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.sld_rot, 1)
        rl.addWidget(self.spn_rot)
        scale_row = QWidget()
        sl = QHBoxLayout(scale_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.sld_scale, 1)
        sl.addWidget(self.spn_scale)
        alpha_row = QWidget()
        al = QHBoxLayout(alpha_row)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(self.sld_tint_alpha, 1)
        al.addWidget(self.spn_tint_alpha)
        tint_row = QWidget()
        tl = QHBoxLayout(tint_row)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.chk_tint)
        tl.addWidget(self.btn_tint_color)
        tl.addWidget(self.chk_tint_strong)
        mask_row = QWidget()
        ml = QHBoxLayout(mask_row)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.sld_mask_offset, 1)
        ml.addWidget(self.spn_mask_offset)
        feather_row = QWidget()
        fl = QHBoxLayout(feather_row)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.sld_feather, 1)
        fl.addWidget(self.spn_feather)

        form.addRow("处理方式", self.cmb_mode)
        form.addRow("旋转(°)", rot_row)
        form.addRow("缩放(%)", scale_row)
        form.addRow("颜色叠加", tint_row)
        form.addRow("颜色透明度(%)", alpha_row)
        form.addRow("掩码腐蚀/膨胀(px)", mask_row)
        form.addRow("羽化(px)", feather_row)

        # 后处理复选框 + 算法选择
        self.chk_harmonize = QCheckBox("色调协调")
        self.chk_harmonize.setToolTip("自动调整素材色调匹配背景")
        self.cmb_harmonize_backend = QComboBox()
        self.cmb_harmonize_backend.setToolTip("色调协调算法")
        self._harmonize_backend_list = get_available_harmonize_backends()
        for backend, display, available in self._harmonize_backend_list:
            self.cmb_harmonize_backend.addItem(display)
            if not available:
                idx = self.cmb_harmonize_backend.count() - 1
                item = self.cmb_harmonize_backend.model().item(idx)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        harm_row = QWidget()
        harm_l = QHBoxLayout(harm_row)
        harm_l.setContentsMargins(0, 0, 0, 0)
        harm_l.addWidget(self.chk_harmonize)
        harm_l.addWidget(self.cmb_harmonize_backend, 1)

        self.chk_seam_repair = QCheckBox("接缝修复")
        self.chk_seam_repair.setToolTip("粘贴后用 inpaint 修复素材边缘接缝")
        self.cmb_inpaint_backend = QComboBox()
        self.cmb_inpaint_backend.setToolTip("接缝修复算法")
        self._inpaint_backend_list = get_available_backends()
        for backend, display, available in self._inpaint_backend_list:
            self.cmb_inpaint_backend.addItem(display)
            if not available:
                idx = self.cmb_inpaint_backend.count() - 1
                item = self.cmb_inpaint_backend.model().item(idx)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        seam_row = QWidget()
        seam_l = QHBoxLayout(seam_row)
        seam_l.setContentsMargins(0, 0, 0, 0)
        seam_l.addWidget(self.chk_seam_repair)
        seam_l.addWidget(self.cmb_inpaint_backend, 1)

        form.addRow("色调协调", harm_row)
        form.addRow("接缝修复", seam_row)

        # ---- 图像调整 GroupBox ----
        grp_adjust = QGroupBox("图像调整")
        adj_form = QFormLayout(grp_adjust)
        # 亮度
        self.sld_brightness = QSlider(Qt.Orientation.Horizontal)
        self.sld_brightness.setRange(-100, 100)
        self.spn_brightness = QSpinBox()
        self.spn_brightness.setRange(-100, 100)
        bright_row = QWidget()
        brl2 = QHBoxLayout(bright_row)
        brl2.setContentsMargins(0, 0, 0, 0)
        brl2.addWidget(self.sld_brightness, 1)
        brl2.addWidget(self.spn_brightness)
        # 对比度
        self.sld_contrast = QSlider(Qt.Orientation.Horizontal)
        self.sld_contrast.setRange(50, 200)
        self.spn_contrast = QSpinBox()
        self.spn_contrast.setRange(50, 200)
        contrast_row = QWidget()
        crl = QHBoxLayout(contrast_row)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.addWidget(self.sld_contrast, 1)
        crl.addWidget(self.spn_contrast)
        # 色相偏移
        self.sld_hue = QSlider(Qt.Orientation.Horizontal)
        self.sld_hue.setRange(-180, 180)
        self.spn_hue = QSpinBox()
        self.spn_hue.setRange(-180, 180)
        hue_row = QWidget()
        hrl = QHBoxLayout(hue_row)
        hrl.setContentsMargins(0, 0, 0, 0)
        hrl.addWidget(self.sld_hue, 1)
        hrl.addWidget(self.spn_hue)
        # 饱和度
        self.sld_sat = QSlider(Qt.Orientation.Horizontal)
        self.sld_sat.setRange(0, 300)
        self.spn_sat = QSpinBox()
        self.spn_sat.setRange(0, 300)
        sat_row = QWidget()
        srl = QHBoxLayout(sat_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.addWidget(self.sld_sat, 1)
        srl.addWidget(self.spn_sat)
        # 高斯模糊
        self.sld_gaussian = QSlider(Qt.Orientation.Horizontal)
        self.sld_gaussian.setRange(0, 50)
        self.spn_gaussian = QSpinBox()
        self.spn_gaussian.setRange(0, 50)
        gauss_row = QWidget()
        grl = QHBoxLayout(gauss_row)
        grl.setContentsMargins(0, 0, 0, 0)
        grl.addWidget(self.sld_gaussian, 1)
        grl.addWidget(self.spn_gaussian)

        adj_form.addRow("亮度", bright_row)
        adj_form.addRow("对比度(%)", contrast_row)
        adj_form.addRow("色相偏移", hue_row)
        adj_form.addRow("饱和度(%)", sat_row)
        adj_form.addRow("高斯模糊(px)", gauss_row)

        grp_bg = QGroupBox("背景与颜色")
        bg_l = QVBoxLayout(grp_bg)
        self.btn_pick_bg_color = QPushButton("背景取色器(点击画面)")
        self.btn_extract_bg_color = QPushButton("一键提取背景主色")
        bg_l.addWidget(self.btn_pick_bg_color)
        bg_l.addWidget(self.btn_extract_bg_color)
        # 颜色调色板（主色按钮）
        self.bg_palette_widget = QWidget()
        self.bg_palette_layout = QHBoxLayout(self.bg_palette_widget)
        self.bg_palette_layout.setContentsMargins(0, 0, 0, 0)
        self.bg_palette_layout.setSpacing(4)
        bg_l.addWidget(self.bg_palette_widget)
        self.bg_palette_buttons: List[QPushButton] = []

        grp_bottom = QGroupBox("背景文件列表")
        btm_l = QVBoxLayout(grp_bottom)
        self.list_bgs = QListWidget()
        btm_l.addWidget(self.list_bgs)

        vl.addWidget(grp_top)
        vl.addWidget(props)
        vl.addWidget(grp_adjust)
        vl.addWidget(grp_bg)
        vl.addWidget(grp_bottom, 1)
        return w

    def _connect_signals(self):
        self.act_load_bg.triggered.connect(self._on_load_bg)
        self.act_load_material.triggered.connect(self._on_load_materials)
        self.act_prev_bg.triggered.connect(lambda: self._switch_bg(-1))
        self.act_next_bg.triggered.connect(lambda: self._switch_bg(1))
        self.act_clear_materials.triggered.connect(self._clear_materials)
        self.act_export.triggered.connect(self._on_export)
        self.act_hq_preview.toggled.connect(self._on_toggle_hq)
        self.act_random_generate.triggered.connect(self._on_random_generate)
        self.act_undo.triggered.connect(self._on_undo)
        self.act_redo.triggered.connect(self._on_redo)

        self.list_materials.itemDoubleClicked.connect(self._on_add_material_from_left)
        self.list_bgs.itemDoubleClicked.connect(self._on_select_bg_from_list)

        self.list_added.currentRowChanged.connect(self._on_added_selection_changed)
        self.btn_delete_added.clicked.connect(self._delete_selected_added)
        self.btn_up.clicked.connect(lambda: self._move_selected(-1))
        self.btn_down.clicked.connect(lambda: self._move_selected(1))
        self.btn_top.clicked.connect(lambda: self._move_selected(-(10**6)))
        self.btn_bottom.clicked.connect(lambda: self._move_selected(10**6))

        self.sld_rot.valueChanged.connect(self.spn_rot.setValue)
        self.spn_rot.valueChanged.connect(self.sld_rot.setValue)
        self.sld_scale.valueChanged.connect(self.spn_scale.setValue)
        self.spn_scale.valueChanged.connect(self.sld_scale.setValue)
        self.sld_tint_alpha.valueChanged.connect(self.spn_tint_alpha.setValue)
        self.spn_tint_alpha.valueChanged.connect(self.sld_tint_alpha.setValue)
        self.sld_mask_offset.valueChanged.connect(self.spn_mask_offset.setValue)
        self.spn_mask_offset.valueChanged.connect(self.sld_mask_offset.setValue)
        self.sld_feather.valueChanged.connect(self.spn_feather.setValue)
        self.spn_feather.valueChanged.connect(self.sld_feather.setValue)
        self.sld_brightness.valueChanged.connect(self.spn_brightness.setValue)
        self.spn_brightness.valueChanged.connect(self.sld_brightness.setValue)
        self.sld_contrast.valueChanged.connect(self.spn_contrast.setValue)
        self.spn_contrast.valueChanged.connect(self.sld_contrast.setValue)
        self.sld_hue.valueChanged.connect(self.spn_hue.setValue)
        self.spn_hue.valueChanged.connect(self.sld_hue.setValue)
        self.sld_sat.valueChanged.connect(self.spn_sat.setValue)
        self.spn_sat.valueChanged.connect(self.sld_sat.setValue)
        self.sld_gaussian.valueChanged.connect(self.spn_gaussian.setValue)
        self.spn_gaussian.valueChanged.connect(self.sld_gaussian.setValue)

        self.sld_rot.valueChanged.connect(self._apply_props_to_item)
        self.sld_scale.valueChanged.connect(self._apply_props_to_item)
        self.cmb_mode.currentIndexChanged.connect(self._apply_props_to_item)
        self.chk_tint.toggled.connect(self._apply_props_to_item)
        self.chk_tint_strong.toggled.connect(self._apply_props_to_item)
        self.sld_tint_alpha.valueChanged.connect(self._apply_props_to_item)
        self.btn_tint_color.clicked.connect(self._choose_tint_color)
        self.sld_mask_offset.valueChanged.connect(self._apply_props_to_item)
        self.sld_feather.valueChanged.connect(self._apply_props_to_item)
        self.sld_brightness.valueChanged.connect(self._apply_props_to_item)
        self.sld_contrast.valueChanged.connect(self._apply_props_to_item)
        self.sld_hue.valueChanged.connect(self._apply_props_to_item)
        self.sld_sat.valueChanged.connect(self._apply_props_to_item)
        self.sld_gaussian.valueChanged.connect(self._apply_props_to_item)
        self.chk_seam_repair.toggled.connect(self._apply_props_to_item)
        self.chk_harmonize.toggled.connect(self._apply_props_to_item)
        self.cmb_harmonize_backend.currentIndexChanged.connect(
            self._on_harmonize_backend_changed,
        )
        self.cmb_inpaint_backend.currentIndexChanged.connect(
            self._on_inpaint_backend_changed,
        )

        self.btn_pick_bg_color.clicked.connect(self._toggle_pick_bg_color)
        self.btn_extract_bg_color.clicked.connect(self._extract_bg_main_color)

        self.scene.selectionChanged.connect(self._on_scene_selection_changed)

        self.act_lasso_fill.toggled.connect(self._on_toggle_lasso_mode)
        self.view.lassoCompleted.connect(self._on_lasso_completed)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.act_lasso_fill.isChecked():
                self.act_lasso_fill.setChecked(False)
                return
        if event.key() == Qt.Key.Key_Delete:
            self._delete_selected_added()
            return
        super().keyPressEvent(event)

    def _on_load_bg(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择背景目录", os.getcwd())
        paths: List[str] = []
        if dir_path:
            # 每次选择新目录前先清空背景列表（不追加）
            self.bg_list.clear()
            self.list_bgs.clear()
            self.current_bg_index = -1
            self.bg_bgr = None
            paths = scan_images_recursively(dir_path)
            # 目录选择成功但没有图片时，再提供文件多选作为补救
            if not paths:
                files, _ = QFileDialog.getOpenFileNames(
                    self,
                    "选择背景图像",
                    os.getcwd(),
                    "图片 (*.png *.jpg *.jpeg *.bmp *.webp)",
                )
                paths = files
        else:
            # 用户取消目录选择时直接返回，不再弹出第二个对话框
            return
        if not paths:
            return
        for p in paths:
            if p not in self.bg_list:
                self.bg_list.append(p)
                item = QListWidgetItem(os.path.basename(p))
                icon = QIcon(p)
                item.setIcon(icon)
                self.list_bgs.addItem(item)
        if self.current_bg_index < 0 and self.bg_list:
            self._set_bg_index(0)
        self._schedule_history_snapshot()

    def _on_load_materials(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择素材目录", os.getcwd())
        pairs: List[Tuple[str, Optional[str]]] = []
        if dir_path:
            # 每次选择新目录前先清空素材列表（不追加）
            self.list_materials.clear()
            pairs = material_pairs_from_dir(dir_path)
            # 目录选择成功但没有图片时，再提供文件多选作为补救
            if not pairs:
                files, _ = QFileDialog.getOpenFileNames(
                    self,
                    "选择素材图像",
                    os.getcwd(),
                    "图片 (*.png *.jpg *.jpeg *.bmp *.webp)",
                )
                pairs = [(p, None) for p in files]
        else:
            # 用户取消目录选择时直接返回，不再弹出第二个对话框
            return
        if not pairs:
            return
        for img_path, mask_path in pairs:
            item = QListWidgetItem(
                os.path.basename(img_path) + ("" if not mask_path else " [+mask]")
            )
            item.setData(Qt.ItemDataRole.UserRole, {"img": img_path, "mask": mask_path})
            icon = QIcon(img_path)
            item.setIcon(icon)
            self.list_materials.addItem(item)

    def _on_add_material_from_left(self, item: QListWidgetItem):
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, dict):
            path = data.get("img")
            mask_path = data.get("mask")
        else:
            path = data
            mask_path = None
        if not isinstance(path, str) or not path:
            QMessageBox.critical(self, "错误", "无效的素材路径。")
            return
        try:
            src = cv_imread_rgba(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            return
        if mask_path:
            mask_img = cv2.imdecode(
                np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            if mask_img is None:
                QMessageBox.warning(
                    self, "警告", f"无法读取掩码: {mask_path}，将忽略该掩码。"
                )
            else:
                src = apply_mask_to_bgra(src, mask_img)
        # 根据 alpha 裁剪到掩码区域，缩小物体框线
        src = crop_to_alpha_bbox(src)
        m = MaterialItem(os.path.basename(path), path, src, self)
        self.scene.addItem(m)
        m.setZValue(len(self.material_items))

        view_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        m.setPos(view_rect.center() - m.boundingRect().center())
        self.material_items.append(m)

        self._rebuild_right_list(select_item=m)
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._push_history()

    def _on_select_bg_from_list(self, item: QListWidgetItem):
        row = self.list_bgs.row(item)
        self._set_bg_index(row)
        self._push_history()

    def _switch_bg(self, step: int):
        if not self.bg_list:
            return
        new_idx = (self.current_bg_index + step) % len(self.bg_list)
        self._set_bg_index(new_idx)
        self._push_history()

    def _set_bg_index(self, idx: int):
        if idx < 0 or idx >= len(self.bg_list):
            return
        self.current_bg_index = idx
        path = self.bg_list[idx]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "警告", f"无法读取背景: {path}")
            return
        self.bg_bgr = img
        qimg = cv_to_qimage_bgra(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
        pix = QPixmap.fromImage(qimg)
        self.bg_pix_item.setPixmap(pix)
        self.scene.setSceneRect(QRectF(QPointF(0, 0), pix.size()))
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self.list_bgs.setCurrentRow(idx)
        # 切换背景时清空所有素材
        self._clear_materials()
        self._disable_hq_overlay()
        self._schedule_hq_preview()

    def _clear_materials(self):
        for m in list(self.material_items):
            self.scene.removeItem(m)
        self.material_items.clear()
        self.list_added.clear()
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._push_history()

    def _on_export(self):
        if self.bg_bgr is None:
            QMessageBox.information(self, "提示", "请先加载背景。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出图像", os.getcwd(), "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return
        canvas = self._render_composite_high_quality(
            include_modes={
                BlendMode.PASTE,
                BlendMode.POISSON_NORMAL,
                BlendMode.POISSON_MIXED,
                BlendMode.KONTEXT_BLEND,
                BlendMode.KONTEXT_HARMONIZE,
            }
        )
        ok = cv2.imwrite(path, canvas)
        if ok:
            QMessageBox.information(self, "完成", "导出成功。")
        else:
            QMessageBox.critical(self, "错误", "导出失败。")

    def _on_random_generate(self):
        """基于当前素材/背景，随机批量生成若干素材实例。"""
        if self.bg_bgr is None:
            QMessageBox.information(self, "提示", "请先加载背景。")
            return
        # 从左侧素材列表收集可用素材名称
        material_defs: List[Dict[str, Optional[str]]] = []
        for i in range(self.list_materials.count()):
            item = self.list_materials.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                img_path = data.get("img")
                mask_path = data.get("mask")
            else:
                img_path = data
                mask_path = None
            if isinstance(img_path, str) and img_path:
                material_defs.append(
                    {
                        "name": os.path.basename(img_path),
                        "img": img_path,
                        "mask": mask_path,
                    }
                )
        if not material_defs:
            QMessageBox.information(self, "提示", "没有可用的素材，请先加载素材。")
            return

        h, w = self.bg_bgr.shape[:2]
        dlg = RandomGenerateDialog(
            [str(m["name"]) for m in material_defs], (w, h), self.bg_bgr, self
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        params = dlg.get_generation_params()
        count = params["count"]
        pos_conf = params["position"]
        x_min, x_max = pos_conf["x_range"]
        y_min, y_max = pos_conf["y_range"]
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)

        blend_conf = params["blend"]
        rotation_conf = params["rotation"]
        scale_conf = params["scale"]
        color_conf = params["color_overlay"]

        def choose_material(idx: int) -> Dict[str, Optional[str]]:
            mode = params["mode"]
            n = len(material_defs)
            if mode == 0:  # 随机选择素材
                return random.choice(material_defs)
            elif mode == 1:  # 使用所有素材（轮询）
                return material_defs[idx % n]
            elif mode == 2:  # 仅使用第一个素材
                return material_defs[0]
            elif mode == 3:  # 均匀分布所有素材
                return material_defs[(idx * max(1, n // max(1, count))) % n]
            return random.choice(material_defs)

        def choose_blend_mode() -> int:
            if not blend_conf["enabled"]:
                return BlendMode.PASTE
            modes = blend_conf.get("modes") or ["normal"]
            m = random.choice(modes)
            if m == "poisson_normal":
                return BlendMode.POISSON_NORMAL
            if m == "poisson_mixed":
                return BlendMode.POISSON_MIXED
            if m == "kontext_blend" and self._kontext_available:
                return BlendMode.KONTEXT_BLEND
            if m == "kontext_harm" and self._kontext_available:
                return BlendMode.KONTEXT_HARMONIZE
            return BlendMode.PASTE

        def choose_color(x: int, y: int) -> Tuple[Tuple[int, int, int], float, bool]:
            """返回 (bgr, alpha, strong_tint)"""
            if not color_conf["enabled"]:
                return (0, 0, 0), 0.0, False
            use_bg = color_conf["use_background_colors"]
            use_preset = color_conf["use_preset"]
            preset_name = color_conf["preset_group"]
            opacity_min, opacity_max = color_conf["opacity_range"]
            alpha = random.uniform(opacity_min, opacity_max)
            if use_bg and self.bg_bgr is not None:
                xx = min(max(0, x), w - 1)
                yy = min(max(0, y), h - 1)
                b, g, r = map(int, self.bg_bgr[yy, xx])
                return (b, g, r), alpha, True
            if use_preset:
                b, g, r = dlg.generate_random_color(preset_name)
                return (b, g, r), alpha, True
            # 完全随机
            b, g, r = dlg.generate_random_color(None)
            return (b, g, r), alpha, False

        # 简单的“尝试避免重叠”：记录放置中心与近似半径，采样位置时做有限次尝试
        placed: List[Tuple[float, float, float]] = []  # (x, y, radius)
        avoid_overlap = bool(pos_conf.get("avoid_overlap", False))

        for i in range(count):
            md = choose_material(i)
            img_path = md["img"]
            mask_path = md.get("mask")
            if not isinstance(img_path, str) or not img_path:
                continue
            try:
                src = cv_imread_rgba(img_path)
            except Exception as e:
                print(f"加载素材失败 {img_path}: {e}")
                continue
            if mask_path:
                mask_img = cv2.imdecode(
                    np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                if mask_img is not None:
                    src = apply_mask_to_bgra(src, mask_img)
            src = crop_to_alpha_bbox(src)

            # 随机旋转/缩放参数
            if rotation_conf["enabled"]:
                rot_min, rot_max = rotation_conf["range"]
                rot = random.uniform(rot_min, rot_max)
            else:
                rot = 0.0
            if scale_conf["enabled"]:
                s_min, s_max = scale_conf["range"]
                scale = random.uniform(s_min, s_max)
            else:
                scale = 1.0

            # 根据缩放后的尺寸估算半径用于防重叠
            sh, sw = src.shape[:2]
            est_w = sw * scale
            est_h = sh * scale
            radius = max(est_w, est_h) * 0.5

            # 选择位置
            attempts = 0
            while True:
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                if not avoid_overlap:
                    break
                ok = True
                for px, py, pr in placed:
                    if (x - px) ** 2 + (y - py) ** 2 < (radius + pr) ** 2:
                        ok = False
                        break
                if ok or attempts > 15:
                    break
                attempts += 1
            placed.append((x, y, radius))

            # 创建素材 item
            m_item = MaterialItem(
                os.path.basename(str(img_path)), str(img_path), src, self
            )
            self.scene.addItem(m_item)
            m_item.setZValue(len(self.material_items))

            # 设置属性
            m_item.set_scale_ratio(scale)
            m_item.set_rotation_deg(int(rot) % 360)
            blend_mode = choose_blend_mode()
            m_item.set_blend_mode(blend_mode)

            bgr, alpha_col, strong = choose_color(int(x), int(y))
            if alpha_col > 0:
                m_item.set_tint(bgr, alpha_col)
                m_item.set_strong_tint(strong)

            # 放置位置（以中心为基准）
            br = m_item.boundingRect()
            m_item.setPos(float(x) - br.width() / 2.0, float(y) - br.height() / 2.0)
            self.material_items.append(m_item)

        self._rebuild_right_list()
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._push_history()

    @staticmethod
    def _project_fg_mask(
        canvas_h: int,
        canvas_w: int,
        src_bgra: np.ndarray,
        cx: int,
        cy: int,
    ) -> np.ndarray:
        """将素材 alpha 通道投射为 canvas 坐标系上的前景 mask。

        Args:
            canvas_h (int): 画布高度.
            canvas_w (int): 画布宽度.
            src_bgra (np.ndarray): 素材 BGRA.
            cx (int): 素材中心 x.
            cy (int): 素材中心 y.

        Returns:
            np.ndarray: canvas 尺寸的 uint8 mask (255=前景).
        """
        sh, sw = src_bgra.shape[:2]
        x0, y0 = cx - sw // 2, cy - sh // 2
        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(canvas_w, x0 + sw), min(canvas_h, y0 + sh)
        fg_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if ix0 >= ix1 or iy0 >= iy1:
            return fg_mask
        sx0, sy0 = ix0 - x0, iy0 - y0
        sx1, sy1 = sx0 + (ix1 - ix0), sy0 + (iy1 - iy0)
        alpha_roi = src_bgra[sy0:sy1, sx0:sx1, 3]
        fg_mask[iy0:iy1, ix0:ix1] = (alpha_roi > 0).astype(np.uint8) * 255
        return fg_mask

    def _render_composite_high_quality(
        self, include_modes: Optional[set] = None
    ) -> np.ndarray:
        """渲染高质量合成结果（含后处理）。"""
        bg = (
            self.bg_bgr
            if self.bg_bgr is not None
            else np.zeros((1, 1, 3), dtype=np.uint8)
        )
        canvas = bg.copy()
        h, w = canvas.shape[:2]
        # 导出按层序：从底到顶
        for m in sorted(self.material_items, key=lambda x: x.zValue()):
            if include_modes is not None and m.blend_mode not in include_modes:
                continue
            pkg = m.to_composite_package()
            src_bgra = pkg["img_bgra"]
            mask = pkg["mask"]
            cx, cy = pkg["center"]
            mode = pkg["mode"]
            tint_color_bgr = tuple(pkg.get("tint_color_bgr", (0, 0, 0)))
            tint_alpha = float(pkg.get("tint_alpha", 0.0))
            do_harmonize = bool(pkg.get("harmonize", False))
            do_seam_repair = bool(pkg.get("seam_repair", False))
            cx = int(max(0, min(w - 1, cx)))
            cy = int(max(0, min(h - 1, cy)))
            if mode == BlendMode.PASTE:
                canvas = self._alpha_paste(canvas, src_bgra, cx, cy)
                # 直接粘贴模式下，如有叠加颜色则在结果上再叠加一次，保证导出一致
                if tint_alpha > 0:
                    sh, sw = src_bgra.shape[:2]
                    x0 = cx - sw // 2
                    y0 = cy - sh // 2
                    x1 = x0 + sw
                    y1 = y0 + sh
                    ix0 = max(0, x0)
                    iy0 = max(0, y0)
                    ix1 = min(w, x1)
                    iy1 = min(h, y1)
                    # 如果没有有效 mask，则跳过颜色叠加
                    if mask is None:
                        continue
                    mask_crop = mask[
                        max(0, -y0) : max(0, -y0) + (iy1 - iy0),
                        max(0, -x0) : max(0, -x0) + (ix1 - ix0),
                    ]
                    self._tint_region_bgr(
                        canvas,
                        ix0,
                        iy0,
                        ix1,
                        iy1,
                        mask_crop,
                        tint_color_bgr,
                        tint_alpha,
                    )
            else:
                # 泊松融合：先根据与背景交集裁剪，避免越界
                sh, sw = src_bgra.shape[:2]
                x0 = cx - sw // 2
                y0 = cy - sh // 2
                x1 = x0 + sw
                y1 = y0 + sh
                ix0 = max(0, x0)
                iy0 = max(0, y0)
                ix1 = min(w, x1)
                iy1 = min(h, y1)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                sx0 = ix0 - x0
                sy0 = iy0 - y0
                sx1 = sx0 + (ix1 - ix0)
                sy1 = sy0 + (iy1 - iy0)
                src_crop = src_bgra[sy0:sy1, sx0:sx1]
                mask_crop = mask[sy0:sy1, sx0:sx1]
                center = ((ix0 + ix1) // 2, (iy0 + iy1) // 2)

                src_bgr = src_crop[:, :, :3]
                mask_255 = (mask_crop > 0).astype(np.uint8) * 255
                flag = (
                    cv2.NORMAL_CLONE
                    if mode == BlendMode.POISSON_NORMAL
                    else cv2.MIXED_CLONE
                )
                try:
                    canvas = cv2.seamlessClone(src_bgr, canvas, mask_255, center, flag)
                except cv2.error:
                    canvas = self._alpha_paste(canvas, src_bgra, cx, cy)
                # 在泊松融合结果上再叠加一次颜色，以确保颜色叠加在泊松模式下也生效
                if tint_alpha > 0:
                    self._tint_region_bgr(
                        canvas,
                        ix0,
                        iy0,
                        ix1,
                        iy1,
                        mask_crop,
                        tint_color_bgr,
                        tint_alpha,
                    )

            # ---- 后处理：色调协调 + 接缝修复 ----
            if do_harmonize or do_seam_repair:
                fg_mask = self._project_fg_mask(h, w, src_bgra, cx, cy)
                if np.any(fg_mask > 0):
                    if do_harmonize:
                        canvas = harmonize_region(canvas, fg_mask)
                    if do_seam_repair:
                        canvas = repair_seam(canvas, fg_mask)
        return canvas

    @staticmethod
    def _alpha_paste(
        dst_bgr: np.ndarray, src_bgra: np.ndarray, cx: int, cy: int
    ) -> np.ndarray:
        out = dst_bgr.copy()
        h, w = dst_bgr.shape[:2]
        sh, sw = src_bgra.shape[:2]
        x0 = cx - sw // 2
        y0 = cy - sh // 2
        x1 = x0 + sw
        y1 = y0 + sh
        ix0 = max(0, x0)
        iy0 = max(0, y0)
        ix1 = min(w, x1)
        iy1 = min(h, y1)
        if ix0 >= ix1 or iy0 >= iy1:
            return out
        sx0 = ix0 - x0
        sy0 = iy0 - y0
        sx1 = sx0 + (ix1 - ix0)
        sy1 = sy0 + (iy1 - iy0)
        src_roi = src_bgra[sy0:sy1, sx0:sx1]
        dst_roi = out[iy0:iy1, ix0:ix1]
        alpha = (src_roi[:, :, 3:4].astype(np.float32)) / 255.0
        dst_roi_f = dst_roi.astype(np.float32)
        src_rgb_f = src_roi[:, :, :3].astype(np.float32)
        blend = src_rgb_f * alpha + dst_roi_f * (1.0 - alpha)
        out[iy0:iy1, ix0:ix1] = np.clip(blend, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _tint_region_bgr(
        canvas: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        mask_gray: np.ndarray,
        tint_color_bgr: Tuple[int, int, int],
        tint_alpha: float,
    ) -> None:
        """在 canvas[y0:y1, x0:x1] 区域内，按 mask * tint_alpha 叠加指定颜色。"""
        if tint_alpha <= 0:
            return
        h, w = canvas.shape[:2]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        if x0 >= x1 or y0 >= y1:
            return
        roi = canvas[y0:y1, x0:x1]
        mask = mask_gray.astype(np.float32) / 255.0
        if mask.shape[:2] != roi.shape[:2]:
            mask = cv2.resize(
                mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        alpha = (mask * float(tint_alpha)).astype(np.float32)[..., None]
        if np.max(alpha) <= 0:
            return
        color = np.array(tint_color_bgr, dtype=np.float32).reshape(1, 1, 3)
        dst = roi.astype(np.float32)
        blended = dst * (1.0 - alpha) + color * alpha
        roi[:, :, :] = np.clip(blended, 0, 255).astype(np.uint8)

    def _on_added_selection_changed(self, row: int):
        if self._suppress_ui:
            return
        for i, m in enumerate(self.material_items):
            m.setSelected(i == row)
        self._populate_props_from_item(self._current_item())

    def _on_scene_selection_changed(self):
        if self._suppress_ui:
            return
        sel = self._current_item()
        if sel is None:
            self.list_added.clearSelection()
            return
        row = self.material_items.index(sel)
        self._suppress_ui = True
        try:
            self.list_added.setCurrentRow(row)
        finally:
            self._suppress_ui = False
        self._populate_props_from_item(sel)

    def _current_item(self) -> Optional[MaterialItem]:
        for m in self.material_items:
            if m.isSelected():
                return m
        if (
            self.material_items
            and self.list_added.currentRow() >= 0
            and self.list_added.currentRow() < len(self.material_items)
        ):
            return self.material_items[self.list_added.currentRow()]
        return None

    def _populate_props_from_item(self, m: Optional[MaterialItem]):
        self._suppress_ui = True
        try:
            if m is None:
                self.cmb_mode.setCurrentIndex(0)
                self.sld_rot.setValue(0)
                self.sld_scale.setValue(100)
                self.chk_tint.setChecked(False)
                self.chk_tint_strong.setChecked(False)
                self.sld_tint_alpha.setValue(0)
                self.sld_mask_offset.setValue(0)
                self.sld_feather.setValue(0)
                self.sld_brightness.setValue(0)
                self.sld_contrast.setValue(100)
                self.sld_hue.setValue(0)
                self.sld_sat.setValue(100)
                self.sld_gaussian.setValue(0)
                self._set_tint_button_color(None)
                return
            self.cmb_mode.setCurrentIndex(m.blend_mode)
            self.sld_rot.setValue(m.rotation_deg)
            self.sld_scale.setValue(int(round(m.scale_ratio * 100)))
            self.chk_tint.setChecked(m.tint_alpha > 0)
            self.sld_tint_alpha.setValue(int(round(m.tint_alpha * 100)))
            self.sld_mask_offset.setValue(int(m.mask_offset))
            self.chk_tint_strong.setChecked(bool(getattr(m, "strong_tint", False)))
            self.sld_feather.setValue(int(m.feather_radius))
            self.sld_brightness.setValue(int(m.brightness))
            self.sld_contrast.setValue(int(m.contrast))
            self.sld_hue.setValue(int(m.hue_shift))
            self.sld_sat.setValue(int(m.saturation))
            self.sld_gaussian.setValue(int(m.gaussian_blur_radius))
            self.chk_seam_repair.setChecked(m.seam_repair)
            self.chk_harmonize.setChecked(m.harmonize)
            self._set_tint_button_color(m.tint_color_bgr if m.tint_alpha > 0 else None)
        finally:
            self._suppress_ui = False

    def _apply_props_to_item(self):
        if self._suppress_ui:
            return
        m = self._current_item()
        if m is None:
            return
        m.set_blend_mode(self.cmb_mode.currentIndex())
        m.set_rotation_deg(self.sld_rot.value())
        m.set_scale_ratio(self.sld_scale.value() / 100.0)
        alpha = (
            (self.sld_tint_alpha.value() / 100.0) if self.chk_tint.isChecked() else 0.0
        )
        m.set_tint(m.tint_color_bgr, alpha)
        # 更新按钮颜色显示
        self._set_tint_button_color(m.tint_color_bgr if alpha > 0 else None)
        m.set_mask_offset(self.sld_mask_offset.value())
        m.set_strong_tint(self.chk_tint_strong.isChecked())
        m.set_feather_radius(self.sld_feather.value())
        m.set_brightness(self.sld_brightness.value())
        m.set_contrast(self.sld_contrast.value())
        m.set_hue_shift(self.sld_hue.value())
        m.set_saturation(self.sld_sat.value())
        m.set_gaussian_blur_radius(self.sld_gaussian.value())
        m.seam_repair = self.chk_seam_repair.isChecked()
        m.harmonize = self.chk_harmonize.isChecked()

        # 若选择了需要 HQ 预览的模式，自动打开
        needs_hq = m.blend_mode != BlendMode.PASTE or m.seam_repair or m.harmonize
        if needs_hq and not self.hq_enabled:
            self.act_hq_preview.setChecked(True)

        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._schedule_history_snapshot()

    def _on_harmonize_backend_changed(self, index: int) -> None:
        """用户切换色调协调算法时更新全局后端设置。"""
        if index < 0 or index >= len(self._harmonize_backend_list):
            return
        backend, display, available = self._harmonize_backend_list[index]
        if not available:
            return
        try:
            set_harmonize_backend(backend)
        except ValueError as e:
            print(f"[MainWindow] 切换色调协调后端失败: {e}")
            return
        print(f"[MainWindow] 色调协调后端 → {display}")
        # 后端变更后刷新 HQ 预览
        self._disable_hq_overlay()
        self._schedule_hq_preview()

    def _on_inpaint_backend_changed(self, index: int) -> None:
        """用户切换接缝修复算法时更新全局后端设置。"""
        if index < 0 or index >= len(self._inpaint_backend_list):
            return
        backend, display, available = self._inpaint_backend_list[index]
        if not available:
            return
        try:
            set_backend(backend)
        except ValueError as e:
            print(f"[MainWindow] 切换接缝修复后端失败: {e}")
            return
        print(f"[MainWindow] 接缝修复后端 → {display}")
        # 后端变更后刷新 HQ 预览
        self._disable_hq_overlay()
        self._schedule_hq_preview()

    def _delete_selected_added(self):
        row = self.list_added.currentRow()
        if row < 0 or row >= len(self.material_items):
            return
        m = self.material_items[row]
        self.scene.removeItem(m)
        del self.material_items[row]
        self._rebuild_right_list()
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._push_history()

    def _choose_tint_color(self):
        m = self._current_item()
        if m is None:
            return
        # 以当前叠加颜色为初始色，避免总是从黑色开始
        if m.tint_alpha > 0:
            b, g, r = m.tint_color_bgr
            init_color = QColor(r, g, b)
        else:
            init_color = QColor(255, 255, 255)
        col = QColorDialog.getColor(init_color, self, "选择叠加颜色")
        if not col.isValid():
            # 不改变
            return
        bgr = (col.blue(), col.green(), col.red())
        # 若当前透明度为 0，则视为用户希望看到效果，自动提升到 100%
        new_alpha = m.tint_alpha if m.tint_alpha > 0 else 0.5
        m.set_tint(bgr, new_alpha)
        # 同步 UI 状态
        self.chk_tint.setChecked(True)
        self.sld_tint_alpha.setValue(int(round(new_alpha * 100)))
        self.spn_tint_alpha.setValue(int(round(new_alpha * 100)))
        self._set_tint_button_color(bgr)
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._schedule_history_snapshot()

    def _toggle_pick_bg_color(self):
        enable = self.btn_pick_bg_color.text().startswith("背景取色器")
        self.view.enable_pick_background_color(enable)
        self.btn_pick_bg_color.setText("退出取色" if enable else "背景取色器(点击画面)")

    def _on_pick_bg_color(self, x: int, y: int):
        self.view.enable_pick_background_color(False)
        self.btn_pick_bg_color.setText("背景取色器(点击画面)")
        if self.bg_bgr is None:
            return
        h, w = self.bg_bgr.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        b, g, r = map(int, self.bg_bgr[y, x])
        m = self._current_item()
        if m is None:
            return
        bgr = (b, g, r)
        if not self.chk_tint.isChecked():
            self.chk_tint.setChecked(True)
        new_alpha = m.tint_alpha if m.tint_alpha > 0 else 1.0
        m.set_tint(bgr, new_alpha)
        self.sld_tint_alpha.setValue(int(round(new_alpha * 100)))
        self.spn_tint_alpha.setValue(int(round(new_alpha * 100)))
        self._set_tint_button_color(bgr)
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._schedule_history_snapshot()

    def _extract_bg_main_color(self):
        if self.bg_bgr is None:
            QMessageBox.information(self, "提示", "请先加载背景。")
            return
        colors = dominant_colors_bgr(self.bg_bgr, k=5)
        if not colors:
            QMessageBox.information(self, "提示", "无法从背景中提取颜色。")
            return
        # 更新调色板按钮
        self._update_bg_palette(colors)
        # 默认将第一主色应用到当前素材（若有）
        m = self._current_item()
        if m is not None:
            b, g, r = colors[0]
            bgr = (b, g, r)
            new_alpha = m.tint_alpha if m.tint_alpha > 0 else 1.0
            m.set_tint(bgr, new_alpha)
            self.chk_tint.setChecked(True)
            self.sld_tint_alpha.setValue(int(round(new_alpha * 100)))
            self.spn_tint_alpha.setValue(int(round(new_alpha * 100)))
            self._set_tint_button_color(bgr)
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._schedule_history_snapshot()

    # ------- 高质量预览（异步生成覆盖图层，仅融合泊松项） -------
    def _on_toggle_hq(self, checked: bool):
        self.hq_enabled = checked
        if not checked:
            self._disable_hq_overlay()
        else:
            self._schedule_hq_preview()

    @staticmethod
    def _needs_hq_preview(m: "MaterialItem") -> bool:
        """判断素材是否需要走 HQ 预览管线。"""
        return m.blend_mode != BlendMode.PASTE or m.seam_repair or m.harmonize

    def _disable_hq_overlay(self):
        self.hq_timer.stop()
        self.hq_overlay_item.setVisible(False)

    def _schedule_hq_preview(self):
        if not self.hq_enabled or self.bg_bgr is None:
            return
        self.hq_timer.start()

    def _run_hq_preview_async(self):
        if not self.hq_enabled or self.bg_bgr is None:
            return
        # 如果上一轮 HQ 任务还在运行，跳过（避免重复提交重量级后处理）
        if self._hq_future is not None and not self._hq_future.done():
            return
        # 只要有任一素材需要 HQ 预览，就把全部素材纳入渲染
        has_hq = any(self._needs_hq_preview(m) for m in self.material_items)
        if not has_hq:
            self._disable_hq_overlay()
            return
        items_state = [m.to_composite_package() for m in self.material_items]
        serial = self.hq_serial = self.hq_serial + 1
        bgr = self.bg_bgr.copy()

        def task(bg_img: np.ndarray, pkgs: List[dict]) -> Tuple[int, np.ndarray]:
            canvas = bg_img.copy()
            h, w = canvas.shape[:2]
            for pkg in sorted(pkgs, key=lambda p: p.get("z", 0)):
                src_bgra = pkg["img_bgra"]
                mask = pkg["mask"]
                mode = pkg["mode"]
                do_harmonize = bool(pkg.get("harmonize", False))
                do_seam_repair = bool(pkg.get("seam_repair", False))
                cx, cy = pkg["center"]
                cx = int(max(0, min(w - 1, cx)))
                cy = int(max(0, min(h - 1, cy)))

                if mode == BlendMode.PASTE:
                    # PASTE 模式进入 HQ 预览仅因为有后处理
                    canvas = MainWindow._alpha_paste(canvas, src_bgra, cx, cy)
                else:
                    # 泊松融合：裁剪避免越界
                    sh, sw = src_bgra.shape[:2]
                    x0 = cx - sw // 2
                    y0 = cy - sh // 2
                    x1, y1 = x0 + sw, y0 + sh
                    ix0, iy0 = max(0, x0), max(0, y0)
                    ix1, iy1 = min(w, x1), min(h, y1)
                    if ix0 >= ix1 or iy0 >= iy1:
                        continue
                    sx0, sy0 = ix0 - x0, iy0 - y0
                    sx1 = sx0 + (ix1 - ix0)
                    sy1 = sy0 + (iy1 - iy0)
                    src_crop = src_bgra[sy0:sy1, sx0:sx1]
                    mask_crop = mask[sy0:sy1, sx0:sx1]
                    center = ((ix0 + ix1) // 2, (iy0 + iy1) // 2)
                    src_bgr = src_crop[:, :, :3]
                    mask_255 = (mask_crop > 0).astype(np.uint8) * 255
                    flag = (
                        cv2.NORMAL_CLONE
                        if mode == BlendMode.POISSON_NORMAL
                        else cv2.MIXED_CLONE
                    )
                    try:
                        canvas = cv2.seamlessClone(
                            src_bgr,
                            canvas,
                            mask_255,
                            center,
                            flag,
                        )
                    except cv2.error:
                        canvas = MainWindow._alpha_paste(
                            canvas,
                            src_bgra,
                            cx,
                            cy,
                        )

                # ---- 后处理 ----
                if do_harmonize or do_seam_repair:
                    fg_mask = MainWindow._project_fg_mask(
                        h,
                        w,
                        src_bgra,
                        cx,
                        cy,
                    )
                    if np.any(fg_mask > 0):
                        if do_harmonize:
                            canvas = harmonize_region(canvas, fg_mask)
                        if do_seam_repair:
                            canvas = repair_seam(canvas, fg_mask)
            return serial, canvas

        # 运行（线程池）
        future = self.executor.submit(task, bgr, items_state)
        self._hq_future = future

        def _done(fut):
            try:
                result_serial, canvas = fut.result()
            except Exception:
                import traceback
                print(f"[HQ Preview] 异步任务失败:\n"
                      f"{traceback.format_exc()}")
                return
            self._hq_signals.finished.emit(result_serial, canvas)

        future.add_done_callback(_done)

    def _apply_hq_result(self, result_serial: int, canvas: np.ndarray):
        if result_serial != self.hq_serial:
            return
        if not self.hq_enabled:
            return
        qimg = cv_to_qimage_bgra(cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA))
        self.hq_overlay_item.setPixmap(QPixmap.fromImage(qimg))
        self.hq_overlay_item.setVisible(True)
        # HQ overlay 的 z-value=9000 在素材之上，视觉上覆盖了素材，
        # 但 overlay 不接收鼠标事件（NoButton），点击会穿透到下方素材。
        # 因此无需改变素材的 opacity / visibility。

    # ------- 颜色调色板与叠加颜色按钮显示 -------
    def _set_tint_button_color(self, bgr: Optional[Tuple[int, int, int]]):
        """将当前叠加颜色显示在按钮背景上；None 时清除样式。"""
        if bgr is None:
            self.btn_tint_color.setStyleSheet("")
        else:
            b, g, r = bgr
            self.btn_tint_color.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"
            )

    def _update_bg_palette(self, colors: List[Tuple[int, int, int]]):
        # 清空旧按钮
        for btn in self.bg_palette_buttons:
            btn.deleteLater()
        self.bg_palette_buttons = []
        # 创建新按钮
        for b, g, r in colors:
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"
            )
            btn.setToolTip(f"R{r} G{g} B{b}")

            def make_slot(color_bgr: Tuple[int, int, int]):
                return lambda _=False, c=color_bgr: self._apply_palette_color(c)

            btn.clicked.connect(make_slot((b, g, r)))
            self.bg_palette_layout.addWidget(btn)
            self.bg_palette_buttons.append(btn)

    def _apply_palette_color(self, color_bgr: Tuple[int, int, int]):
        m = self._current_item()
        if m is None:
            return
        if not self.chk_tint.isChecked():
            self.chk_tint.setChecked(True)
        # 若当前透明度为 0，则默认设为 1.0（100%），并同步 UI
        alpha = m.tint_alpha if m.tint_alpha > 0 else 1.0
        m.set_tint(color_bgr, alpha)
        self.sld_tint_alpha.setValue(int(round(alpha * 100)))
        self.spn_tint_alpha.setValue(int(round(alpha * 100)))
        self._set_tint_button_color(color_bgr)
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._schedule_history_snapshot()

    # ------- 套索填充（内容识别） -------
    def _on_toggle_lasso_mode(self, checked: bool):
        """进入/退出套索选区模式，与背景取色互斥。"""
        if checked:
            # 退出取色器模式（互斥）
            self.view.enable_pick_background_color(False)
            self.btn_pick_bg_color.setText("背景取色器(点击画面)")
        self.view.enable_lasso_mode(checked)

    def _on_lasso_completed(self, points: list):
        """套索选区闭合后的处理入口。"""
        if self.bg_bgr is None:
            QMessageBox.warning(self, "警告", "请先加载背景图片。")
            self.view.clear_lasso_path()
            self.act_lasso_fill.setChecked(False)
            return

        # 将 QPointF 列表转换为 numpy 整数坐标
        pts = np.array([[int(p.x()), int(p.y())] for p in points], dtype=np.int32)

        # 弹出参数对话框
        dlg = ContentAwareFillDialog(self)
        params = dlg.get_params()

        # 清除套索可视化并退出套索模式
        self.view.clear_lasso_path()
        self.act_lasso_fill.setChecked(False)

        if params is None:
            return

        self._apply_content_aware_fill(pts, params)

    def _apply_content_aware_fill(self, pts: np.ndarray, params: Dict[str, Any]):
        """对背景图执行 PatchMatch 内容识别填充（异步，不阻塞 UI）。

        Args:
            pts (np.ndarray): 多边形顶点数组，shape (N, 2)，dtype int32。
            params (dict): 填充参数，包含 patch_size / expand / backend。
        """
        # 应用用户选择的后端
        backend: InpaintBackend = params.get("backend", InpaintBackend.AUTO)
        set_backend(backend)

        # 在修改前保存背景快照到历史（用于撤销）
        state = self._capture_state()
        state["bg_bgr_snapshot"] = self.bg_bgr.copy()
        # 截断重做链并推入
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
        self.history.append(state)
        if len(self.history) > 50:
            self.history = self.history[-50:]
        self.history_index = len(self.history) - 1

        h, w = self.bg_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # 选区扩展（膨胀 mask）
        if params["expand"] > 0:
            k = params["expand"]
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        # 异步执行内容识别填充
        backend_name = _get_backend_name()
        print(f"[Content-Aware Fill] 使用后端: {backend_name}")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(f"内容识别填充中 ({backend_name})…")

        # 禁用套索操作，防止重复提交
        self.act_lasso_fill.setEnabled(False)

        img_copy = self.bg_bgr.copy()
        patch_size = params["patch_size"]
        max_size = params.get("max_size", 0)
        signals = self._inpaint_signals

        def _task():
            """在线程池中执行填充。"""
            try:
                result = patchmatch_inpaint(
                    img_copy, mask, patch_size=patch_size, max_size=max_size
                )
                signals.finished.emit(result)
            except Exception as exc:
                signals.error.emit(str(exc))

        self._inpaint_future = self.executor.submit(_task)

    def _on_inpaint_finished(self, result: np.ndarray):
        """内容识别填充完成回调（主线程）。

        Args:
            result (np.ndarray): 填充结果 BGR uint8.
        """
        QApplication.restoreOverrideCursor()
        self.act_lasso_fill.setEnabled(True)
        self.statusBar().showMessage("内容识别填充完成", 3000)
        self._inpaint_future = None

        self.bg_bgr = result
        self._refresh_bg_display()
        self._disable_hq_overlay()
        self._schedule_hq_preview()

    def _on_inpaint_error(self, msg: str):
        """内容识别填充失败回调（主线程）。

        Args:
            msg (str): 错误信息.
        """
        QApplication.restoreOverrideCursor()
        self.act_lasso_fill.setEnabled(True)
        self.statusBar().showMessage("内容识别填充失败", 3000)
        self._inpaint_future = None
        QMessageBox.critical(self, "内容识别填充失败", msg)

    def _refresh_bg_display(self):
        """根据当前 bg_bgr 刷新背景显示（不从文件重新加载）。"""
        if self.bg_bgr is None:
            return
        qimg = cv_to_qimage_bgra(cv2.cvtColor(self.bg_bgr, cv2.COLOR_BGR2BGRA))
        pix = QPixmap.fromImage(qimg)
        self.bg_pix_item.setPixmap(pix)

    # ------- 图层顺序控制 -------
    def _rebuild_right_list(self, select_item: Optional[MaterialItem] = None):
        self.list_added.clear()
        for m in self.material_items:
            item = QListWidgetItem(m.name)
            item.setData(Qt.ItemDataRole.UserRole, m)
            self.list_added.addItem(item)
        if select_item is not None:
            try:
                idx = self.material_items.index(select_item)
                self.list_added.setCurrentRow(idx)
            except ValueError:
                pass

    def _move_selected(self, delta: int):
        row = self.list_added.currentRow()
        if row < 0 or row >= len(self.material_items):
            return
        new_row = max(0, min(len(self.material_items) - 1, row + delta))
        if new_row == row:
            return
        m = self.material_items.pop(row)
        self.material_items.insert(new_row, m)
        # 重设 z 值 与 列表
        for i, it in enumerate(self.material_items):
            it.setZValue(i)
        self._rebuild_right_list(select_item=m)
        self._disable_hq_overlay()
        self._schedule_hq_preview()
        self._push_history()

    # ------- 历史（撤销/重做） -------
    def _capture_state(self) -> Dict[str, Any]:
        return {
            "bg_index": int(self.current_bg_index),
            "materials": [m.to_state_dict() for m in self.material_items],
        }

    def _restore_state(self, state: Dict[str, Any]):
        # 清空现有
        for m in list(self.material_items):
            self.scene.removeItem(m)
        self.material_items.clear()
        self.list_added.clear()
        # 背景（优先从快照恢复，否则从文件加载）
        bg_snapshot = state.get("bg_bgr_snapshot", None)
        if bg_snapshot is not None:
            self.bg_bgr = bg_snapshot.copy()
            self._refresh_bg_display()
        else:
            bg_idx = state.get("bg_index", -1)
            if 0 <= bg_idx < len(self.bg_list):
                self._set_bg_index(bg_idx)
        # 材质（先按 z 排序，再按索引保证稳定顺序）
        materials = state.get("materials", [])
        materials_sorted = sorted(materials, key=lambda s: s.get("z", 0))
        for i, sd in enumerate(materials_sorted):
            path = sd.get("path", "")
            name = sd.get("name", os.path.basename(path) if path else f"Item{i}")
            try:
                img = cv_imread_rgba(path)
            except Exception:
                continue
            m = MaterialItem(name, path, img, self)
            self.scene.addItem(m)
            m.setZValue(i)
            m.setPos(
                QPointF(
                    float(sd.get("pos", (0.0, 0.0))[0]),
                    float(sd.get("pos", (0.0, 0.0))[1]),
                )
            )
            m.set_scale_ratio(float(sd.get("scale", 1.0)))
            m.set_rotation_deg(int(sd.get("rotation", 0)))
            color = tuple(sd.get("tint_color_bgr", (0, 0, 0)))
            alpha = float(sd.get("tint_alpha", 0.0))
            m.set_tint(color, alpha)
            m.set_blend_mode(int(sd.get("mode", BlendMode.PASTE)))
            m.set_mask_offset(int(sd.get("mask_offset", 0)))
            m.set_strong_tint(bool(sd.get("strong_tint", False)))
            m.set_feather_radius(int(sd.get("feather_radius", 0)))
            m.set_brightness(int(sd.get("brightness", 0)))
            m.set_contrast(int(sd.get("contrast", 100)))
            m.set_hue_shift(int(sd.get("hue_shift", 0)))
            m.set_saturation(int(sd.get("saturation", 100)))
            m.set_gaussian_blur_radius(int(sd.get("gaussian_blur_radius", 0)))
            m.seam_repair = bool(sd.get("seam_repair", False))
            m.harmonize = bool(sd.get("harmonize", False))
            self.material_items.append(m)
        self._rebuild_right_list()
        self._disable_hq_overlay()
        self._schedule_hq_preview()

    def _push_history(self):
        state = self._capture_state()
        # 与当前顶层相同则不追加
        if (
            0 <= self.history_index < len(self.history)
            and self.history[self.history_index] == state
        ):
            return
        # 截断重做链
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
        self.history.append(state)
        # 限制长度
        if len(self.history) > 50:
            self.history = self.history[-50:]
        self.history_index = len(self.history) - 1

    def _on_undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self._restore_state(self.history[self.history_index])

    def _on_redo(self):
        if self.history_index + 1 < len(self.history):
            self.history_index += 1
            self._restore_state(self.history[self.history_index])

    def _schedule_history_snapshot(self):
        self.hist_timer.start()

    # ------- 交互协作 -------
    def _on_item_interaction_started(self, m: MaterialItem):
        self._disable_hq_overlay()

    def _on_item_interaction_finished(self, m: MaterialItem):
        self._schedule_history_snapshot()
        self._schedule_hq_preview()

    def _sync_rotation_from_item(self, m: MaterialItem) -> None:
        """从画布旋转手柄拖拽时同步到属性面板。"""
        if m != self._current_item():
            return
        self._suppress_ui = True
        try:
            self.sld_rot.setValue(m.rotation_deg)
            self.spn_rot.setValue(m.rotation_deg)
        finally:
            self._suppress_ui = False


def print_basic_usage():
    """启动时在终端输出一次基本操作说明。"""
    print(
        """
========== Material Editor v2 基本操作 ==========
启动方式：
  在项目目录下执行：python main.py

界面说明：
  左侧：素材列表（双击素材加入画布）
  中间：画布预览（滚轮缩放，按住左键拖动画布，点击/拖动素材）
  右上：已添加素材 + 属性（旋转、缩放、颜色叠加、掩码腐蚀/膨胀、混合模式）
  右下：背景列表 + 背景取色器 / 一键提取背景主色

工具栏常用按钮：
  - 加载背景：选择背景目录或图片文件
  - 加载素材：选择素材目录，支持 A.png + A_mask.png 掩码对
  - 上一张 / 下一张：切换背景（会清空当前画布素材）
  - 清空素材：移除所有素材
  - 一键导出：导出当前合成结果
  - 随机生成素材：按配置批量随机布置素材
  - 高质量预览：启用泊松融合高质量预览
  - 撤销 / 重做：Ctrl+Z / Ctrl+Y

提示：
  - 泊松融合 Normal / Mix 模式建议配合“高质量预览”使用。
  - 颜色叠加可配合“强叠加模式”和透明度滑条调节效果。
============================================
""".strip()
    )


def main():
    print_basic_usage()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
