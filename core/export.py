"""
导出功能模块
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.layer import LayerManager
from core.material import MaterialInstance


class ImageComposer:
    """图像合成器"""

    def __init__(self):
        self.clone_mode = cv2.NORMAL_CLONE

    def set_clone_mode(self, mode: str):
        """设置合成模式"""
        mode_map = {
            "NORMAL": cv2.NORMAL_CLONE,
            "MIXED": cv2.MIXED_CLONE,
            "COPY": "COPY",  # 自定义复制模式
        }
        self.clone_mode = mode_map.get(mode, cv2.NORMAL_CLONE)

    def compose_image(
        self, background: np.ndarray, instances: List[MaterialInstance]
    ) -> np.ndarray:
        """合成图像"""
        result = background.copy()

        # 按图层顺序合成（从底层到顶层）
        for instance in instances:
            if not instance.visible:
                continue

            try:
                result = self._blend_instance(result, instance)
            except Exception as e:
                print(f"合成素材失败: {e}")
                continue

        return result

    def _blend_instance(
        self, background: np.ndarray, instance: MaterialInstance
    ) -> np.ndarray:
        """将素材实例混合到背景中"""
        transformed_image = instance.get_transformed_image()
        transformed_mask = instance.get_transformed_mask()

        x1, y1, x2, y2 = instance.get_bounding_rect()

        # 确保边界在背景图像范围内
        bg_h, bg_w = background.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(bg_w, x2)
        y2 = min(bg_h, y2)

        if x1 >= x2 or y1 >= y2:
            return background

        # 计算素材在变换图像中的对应区域
        mat_x1 = (
            max(0, -instance.get_bounding_rect()[0])
            if instance.get_bounding_rect()[0] < 0
            else 0
        )
        mat_y1 = (
            max(0, -instance.get_bounding_rect()[1])
            if instance.get_bounding_rect()[1] < 0
            else 0
        )
        mat_x2 = mat_x1 + (x2 - x1)
        mat_y2 = mat_y1 + (y2 - y1)

        # 裁剪素材和掩码
        material_roi = transformed_image[mat_y1:mat_y2, mat_x1:mat_x2]

        if transformed_mask is not None:
            mask_roi = transformed_mask[mat_y1:mat_y2, mat_x1:mat_x2]

            if self.clone_mode == "COPY":
                # 直接复制模式
                background_roi = background[y1:y2, x1:x2]
                gray_mask = (
                    cv2.cvtColor(mask_roi, cv2.COLOR_BGR2GRAY)
                    if len(mask_roi.shape) == 3
                    else mask_roi
                )
                _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
                background_roi[binary_mask > 0] = material_roi[binary_mask > 0]
            else:
                # 使用seamlessClone
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2

                # 调整大小以匹配
                if material_roi.shape[:2] != mask_roi.shape[:2]:
                    mask_roi = cv2.resize(
                        mask_roi, (material_roi.shape[1], material_roi.shape[0])
                    )

                try:
                    background = cv2.seamlessClone(
                        material_roi,
                        background,
                        mask_roi,
                        (center_x, center_y),
                        self.clone_mode,
                    )
                except:
                    # 如果seamlessClone失败，使用直接复制
                    background_roi = background[y1:y2, x1:x2]
                    gray_mask = (
                        cv2.cvtColor(mask_roi, cv2.COLOR_BGR2GRAY)
                        if len(mask_roi.shape) == 3
                        else mask_roi
                    )
                    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
                    background_roi[binary_mask > 0] = material_roi[binary_mask > 0]
        else:
            # 没有掩码，直接覆盖
            background[y1:y2, x1:x2] = material_roi

        return background


class ProjectExporter:
    """项目导出器"""

    def __init__(self):
        self.composer = ImageComposer()

    def export_project(
        self,
        background: np.ndarray,
        layer_manager: LayerManager,
        output_path: Path,
        project_name: str = "project",
        export_image: bool = True,
        export_json: bool = True,
    ) -> bool:
        """导出项目"""
        try:
            instances = layer_manager.get_all_instances()

            if export_image:
                # 合成并保存图像
                composed_image = self.composer.compose_image(background, instances)
                image_path = output_path / f"{project_name}.jpg"
                cv2.imencode(".jpg", composed_image)[1].tofile(image_path.as_posix())

            if export_json:
                # 生成并保存标注文件
                annotation_data = self._generate_annotation_data(
                    background, instances, f"{project_name}.jpg"
                )
                json_path = output_path / f"{project_name}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(annotation_data, f, indent=4, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"导出失败: {e}")
            return False

    def _generate_annotation_data(
        self, background: np.ndarray, instances: List[MaterialInstance], image_name: str
    ) -> Dict[str, Any]:
        """生成标注数据"""
        bg_h, bg_w = background.shape[:2]
        bg_c = background.shape[2] if len(background.shape) == 3 else 1

        annotation = {
            "info": {
                "description": "Material Editor",
                "folder": "",
                "name": image_name,
                "width": bg_w,
                "height": bg_h,
                "depth": bg_c,
                "note": "",
            },
            "object": [],
        }

        for instance in instances:
            if instance.visible:
                annotation["object"].append(instance.to_dict())

        return annotation

    def export_layers_separately(
        self,
        background: np.ndarray,
        layer_manager: LayerManager,
        output_path: Path,
        project_name: str = "project",
    ) -> bool:
        """分别导出各图层"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)

            for layer in layer_manager.layers:
                if not layer.visible or not layer.instances:
                    continue

                # 合成该图层
                composed_image = self.composer.compose_image(
                    background, layer.instances
                )
                layer_path = (
                    output_path
                    / f"{project_name}_layer_{layer.layer_id}_{layer.name}.jpg"
                )
                cv2.imencode(".jpg", composed_image)[1].tofile(layer_path.as_posix())

                # 导出该图层的标注
                annotation_data = self._generate_annotation_data(
                    background, layer.instances, layer_path.name
                )
                json_path = (
                    output_path
                    / f"{project_name}_layer_{layer.layer_id}_{layer.name}.json"
                )
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(annotation_data, f, indent=4, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"分层导出失败: {e}")
            return False

    def export_image(
        self, layer_manager: LayerManager, background: np.ndarray, output_path: str
    ) -> bool:
        """导出图像"""
        try:
            instances = layer_manager.get_all_instances()
            composed_image = self.composer.compose_image(background, instances)

            # 保存图像
            success = cv2.imwrite(output_path, composed_image)
            return success
        except Exception as e:
            print(f"导出图像失败: {e}")
            return False

    def export_annotation(self, layer_manager: LayerManager, output_path: str) -> bool:
        """导出标注文件"""
        try:
            import json
            from pathlib import Path

            # 创建标注数据
            annotation_data = {"version": "1.0", "annotations": []}

            instances = layer_manager.get_all_instances()
            for instance in instances:
                if instance.visible:
                    annotation_data["annotations"].append(instance.to_dict())

            # 保存JSON文件
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(annotation_data, f, indent=4, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"导出标注失败: {e}")
            return False
