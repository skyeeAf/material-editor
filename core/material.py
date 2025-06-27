"""
素材管理模块
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MaterialInfo:
    """素材信息"""

    path: Path
    name: str
    image: np.ndarray
    mask: Optional[np.ndarray] = None
    category: str = "material"
    category_id: int = 0

    @property
    def image_path(self) -> str:
        """获取图像路径字符串"""
        return str(self.path)


class MaterialInstance:
    """素材实例 - 代表画布上的一个素材对象"""

    def __init__(
        self,
        material_info: MaterialInfo,
        x: int,
        y: int,
        scale: float = 1.0,
        rotation: float = 0.0,
        layer_id: int = 0,
        blend_mode: str = "normal",  # 混合模式: normal, poisson_normal, poisson_mixed
        color_overlay: Optional[Tuple[int, int, int]] = None,  # RGB颜色叠加
        overlay_opacity: float = 0.0,  # 叠加透明度 (0.0-1.0)
    ):
        self.material_info = material_info
        self.x = x  # 中心点x坐标
        self.y = y  # 中心点y坐标
        self.scale = scale
        self.rotation = rotation  # 旋转角度（度）
        self.layer_id = layer_id
        self.selected = False
        self.visible = True
        self.blend_mode = blend_mode  # 混合模式

        # 色彩叠加属性
        self.color_overlay = color_overlay  # RGB颜色 (r, g, b)
        self.overlay_opacity = overlay_opacity  # 叠加透明度

        # 缓存变换后的图像和掩码
        self._transformed_image_cache: Optional[np.ndarray] = None
        self._transformed_mask_cache: Optional[np.ndarray] = None
        self._transform_cache_key: Optional[Tuple[float, float]] = None

        # 泊松融合结果缓存
        self._poisson_cache: Optional[Dict[str, Any]] = None
        self._poisson_cache_key: Optional[Tuple[int, int, str]] = None

        # 实际显示边界框（用于泊松融合等可能改变显示区域的混合模式）
        self._actual_display_bbox: Optional[Tuple[int, int, int, int]] = None

    @property
    def material_name(self) -> str:
        """获取素材名称"""
        return self.material_info.name

    def get_transformed_image(self) -> np.ndarray:
        """获取变换后的图像（包含色彩叠加）"""
        if self._transformed_image_cache is None:
            self._update_transformed_cache()

        # 应用色彩叠加
        if self.color_overlay and self.overlay_opacity > 0.0:
            return self._apply_color_overlay(self._transformed_image_cache)
        else:
            return self._transformed_image_cache.copy()

    def _get_transformed_image_without_color(self) -> np.ndarray:
        """获取不包含色彩叠加的变换图像（用于泊松融合）"""
        if self._transformed_image_cache is None:
            self._update_transformed_cache()
        return self._transformed_image_cache.copy()

    def get_transformed_mask(self) -> Optional[np.ndarray]:
        """获取变换后的掩码"""
        if self._transformed_mask_cache is None:
            self._update_transformed_cache()
        return self._transformed_mask_cache

    def _update_transformed_cache(self):
        """更新变换缓存"""
        # 检查是否需要更新缓存
        cache_key = (self.scale, self.rotation)
        if (
            self._transform_cache_key == cache_key
            and self._transformed_image_cache is not None
        ):
            return

        # 变换图像（不包含色彩叠加）
        self._transformed_image_cache = self._transform_image(self.material_info.image)

        # 变换掩码
        if self.material_info.mask is not None:
            self._transformed_mask_cache = self._transform_image(
                self.material_info.mask
            )
        else:
            self._transformed_mask_cache = None

        # 更新缓存键
        self._transform_cache_key = cache_key

        # 清除泊松融合缓存（因为变换改变了）
        self._poisson_cache = None
        self._poisson_cache_key = None

    def _apply_color_overlay(self, image: np.ndarray) -> np.ndarray:
        """应用色彩叠加"""
        if self.color_overlay is None or self.overlay_opacity <= 0.0:
            return image

        # 创建颜色叠加图层
        overlay = np.full_like(image, self.color_overlay[::-1])  # BGR格式

        # 应用透明度混合
        alpha = self.overlay_opacity
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return result

    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        """对图像进行变换（缩放和旋转）"""
        if self.scale == 1.0 and self.rotation == 0.0:
            return image.copy()

        result = image.copy()

        # 先缩放
        if self.scale != 1.0:
            h, w = result.shape[:2]
            new_h, new_w = int(h * self.scale), int(w * self.scale)
            result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 再旋转
        if self.rotation != 0.0:
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, -self.rotation, 1.0)

            # 计算旋转后的边界框
            cos_val = np.abs(matrix[0, 0])
            sin_val = np.abs(matrix[0, 1])
            new_w = int((h * sin_val) + (w * cos_val))
            new_h = int((h * cos_val) + (w * sin_val))

            # 调整变换矩阵以包含完整的旋转图像
            matrix[0, 2] += (new_w / 2) - center[0]
            matrix[1, 2] += (new_h / 2) - center[1]

            border_val = (
                (0, 0, 0, 0)
                if len(result.shape) == 3 and result.shape[2] == 4
                else (0, 0, 0)
            )
            result = cv2.warpAffine(
                result, matrix, (new_w, new_h), borderValue=border_val
            )

        return result

    def get_bounding_rect(self) -> Tuple[int, int, int, int]:
        """获取边界框 (x1, y1, x2, y2) - 基于完整变换图像"""
        transformed = self.get_transformed_image()
        h, w = transformed.shape[:2]

        x1 = self.x - w // 2
        y1 = self.y - h // 2
        x2 = x1 + w
        y2 = y1 + h

        return x1, y1, x2, y2

    def get_mask_bounding_rect(self) -> Tuple[int, int, int, int]:
        """获取基于掩码的边界框 (x1, y1, x2, y2)"""
        transformed_mask = self.get_transformed_mask()

        if transformed_mask is None:
            # 如果没有掩码，返回完整的图像边界框
            return self.get_bounding_rect()

        # 转换掩码为灰度图
        if len(transformed_mask.shape) == 3:
            gray_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY)
        else:
            gray_mask = transformed_mask.copy()

        # 创建二值掩码
        _, binary_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # 如果没有找到轮廓，返回完整的图像边界框
            return self.get_bounding_rect()

        # 计算所有轮廓的总边界框
        x_coords = []
        y_coords = []

        for contour in contours:
            for point in contour:
                x_coords.append(point[0][0])
                y_coords.append(point[0][1])

        if not x_coords or not y_coords:
            return self.get_bounding_rect()

        mask_x1 = min(x_coords)
        mask_y1 = min(y_coords)
        mask_x2 = max(x_coords)
        mask_y2 = max(y_coords)

        # 使用不包含色彩叠加的变换图像来计算尺寸，确保与掩码尺寸一致
        transformed_image = self._get_transformed_image_without_color()
        h, w = transformed_image.shape[:2]

        # 变换后图像的左上角在画布中的位置
        image_left = self.x - w // 2
        image_top = self.y - h // 2

        # 掩码边界框在画布中的实际位置
        x1 = image_left + mask_x1
        y1 = image_top + mask_y1
        x2 = image_left + mask_x2
        y2 = image_top + mask_y2

        return x1, y1, x2, y2

    def contains_point(self, px: int, py: int) -> bool:
        """检查点是否在素材内"""
        x1, y1, x2, y2 = self.get_bounding_rect()
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return False

        # 如果有掩码，检查掩码
        transformed_mask = self.get_transformed_mask()
        if transformed_mask is not None:
            mask_x = px - x1
            mask_y = py - y1
            if (
                0 <= mask_x < transformed_mask.shape[1]
                and 0 <= mask_y < transformed_mask.shape[0]
            ):
                mask_pixel = transformed_mask[mask_y, mask_x]
                # 检查掩码值（假设白色为有效区域）
                return bool(np.any(mask_pixel > 128))

        return True

    def get_segment_points(self) -> List[Dict[str, int]]:
        """获取分割点坐标（用于标注）"""
        transformed_mask = self.get_transformed_mask()
        if transformed_mask is None:
            # 如果没有掩码，返回边界框的四个角点
            x1, y1, x2, y2 = self.get_bounding_rect()
            return [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x2, "y": y2},
                {"x": x1, "y": y2},
            ]

        # 从掩码中提取轮廓
        gray = (
            cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY)
            if len(transformed_mask.shape) == 3
            else transformed_mask
        )

        # 创建二值掩码，使用更合适的阈值
        _, binary_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )

        if not contours:
            # 如果没有找到轮廓，返回图像边界框的四个角点
            x1, y1, x2, y2 = self.get_bounding_rect()
            return [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x2, "y": y2},
                {"x": x1, "y": y2},
            ]

        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 简化轮廓 - 使用更精细的参数来保留更多细节
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 转换为绝对坐标 - 使用不包含色彩叠加的变换图像来计算尺寸，确保与掩码尺寸一致
        transformed_image = self._get_transformed_image_without_color()
        h, w = transformed_image.shape[:2]

        # 变换后图像的左上角在画布中的位置
        image_left = self.x - w // 2
        image_top = self.y - h // 2

        points = []
        for point in simplified:
            px, py = point[0]
            # 掩码坐标转换为画布坐标
            points.append({"x": int(px + image_left), "y": int(py + image_top)})

        return points

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于导出）"""
        x1, y1, x2, y2 = self.get_bounding_rect()

        return {
            "category": self.material_info.category,
            "category_id": self.material_info.category_id,
            "group": 0,
            "segment": self.get_segment_points(),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": x2 - x1,
                "h": y2 - y1,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            },
            "layer": self.layer_id,
            "note": "",
            "blend_mode": self.blend_mode,
        }

    def get_display_bounding_rect(self) -> Tuple[int, int, int, int]:
        """获取显示边界框 - 根据混合模式返回最合适的边界框"""
        # 如果有实际显示边界框（通常用于泊松融合），优先使用
        if self._actual_display_bbox is not None:
            return self._actual_display_bbox

        # 否则根据混合模式选择合适的边界框
        if self.blend_mode in ["poisson_normal", "poisson_mixed"]:
            # 泊松融合可能会稍微扩展边界，使用掩码边界框并稍微扩展
            x1, y1, x2, y2 = self.get_mask_bounding_rect()
            expansion = 2  # 像素
            return (
                max(0, x1 - expansion),
                max(0, y1 - expansion),
                x2 + expansion,
                y2 + expansion,
            )
        else:
            # 普通模式使用掩码边界框
            return self.get_mask_bounding_rect()


class MaterialManager:
    """素材管理器"""

    def __init__(self):
        self.materials: Dict[str, MaterialInfo] = {}

    def load_materials_from_directory(self, material_dir: str) -> int:
        """从目录加载素材"""
        return self.load_material_from_directory(Path(material_dir))

    def load_material_from_directory(
        self, material_dir: Path, suffix: str = ".png"
    ) -> int:
        """从目录加载素材"""
        loaded_count = 0

        for material_path in material_dir.rglob(f"*{suffix}"):
            if "mask" in material_path.name:
                continue

            # 加载素材图像
            image = cv2.imdecode(
                np.fromfile(material_path.as_posix(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                continue

            # 查找对应的掩码文件
            mask_path = material_path.with_name(material_path.stem + "_mask.png")
            mask = None
            if mask_path.exists():
                mask = cv2.imdecode(
                    np.fromfile(mask_path.as_posix(), dtype=np.uint8), cv2.IMREAD_COLOR
                )

            # 创建素材信息
            material_info = MaterialInfo(
                path=material_path,
                name=material_path.stem,
                image=image,
                mask=mask,
                category=material_path.parent.name
                if material_path.parent.name != material_dir.name
                else "material",
            )

            self.materials[material_info.name] = material_info
            loaded_count += 1

        return loaded_count

    def get_material_names(self) -> List[str]:
        """获取所有素材名称"""
        return list(self.materials.keys())

    def get_material(self, name: str) -> Optional[MaterialInfo]:
        """获取指定名称的素材"""
        return self.materials.get(name)

    def create_instance(
        self,
        material_name: str,
        x: int,
        y: int,
        scale: float = 1.0,
        rotation: float = 0.0,
        layer_id: int = 0,
        blend_mode: str = "normal",
        color_overlay: Optional[Tuple[int, int, int]] = None,
        overlay_opacity: float = 0.0,
    ) -> Optional[MaterialInstance]:
        """创建素材实例"""
        material_info = self.get_material(material_name)
        if material_info is None:
            return None

        return MaterialInstance(
            material_info,
            x,
            y,
            scale,
            rotation,
            layer_id,
            blend_mode,
            color_overlay,
            overlay_opacity,
        )
