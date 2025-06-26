"""
图层管理模块
"""

from typing import Any, Dict, List, Optional

from core.material import MaterialInstance


class Layer:
    """图层类"""

    def __init__(self, layer_id: int, name: str = ""):
        self.layer_id = layer_id
        self.name = name or f"图层 {layer_id}"
        self.visible = True
        self.locked = False
        self.instances: List[MaterialInstance] = []

    def add_instance(self, instance: MaterialInstance):
        """添加素材实例到图层"""
        instance.layer_id = self.layer_id
        self.instances.append(instance)

    def remove_instance(self, instance: MaterialInstance):
        """从图层移除素材实例"""
        if instance in self.instances:
            self.instances.remove(instance)

    def get_instances_at_point(self, x: int, y: int) -> List[MaterialInstance]:
        """获取指定点的所有素材实例"""
        result = []
        for instance in self.instances:
            if instance.visible and instance.contains_point(x, y):
                result.append(instance)
        return result

    def clear(self):
        """清空图层"""
        self.instances.clear()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "layer_id": self.layer_id,
            "name": self.name,
            "visible": self.visible,
            "locked": self.locked,
            "instances": [instance.to_dict() for instance in self.instances],
        }


class LayerManager:
    """图层管理器"""

    def __init__(self):
        self.layers: List[Layer] = []
        self.current_layer_id = 1  # 设置为1，与第一个图层ID匹配
        self._next_layer_id = 1

        # 创建默认图层
        default_layer = self.add_layer("默认图层")
        self.current_layer_id = default_layer.layer_id  # 确保当前图层ID正确

    def add_layer(self, name: str = "") -> Layer:
        """添加新图层"""
        layer = Layer(self._next_layer_id, name)
        self.layers.append(layer)
        self._next_layer_id += 1
        return layer

    def remove_layer(self, layer_id: int) -> bool:
        """删除图层"""
        if len(self.layers) <= 1:  # 至少保留一个图层
            return False

        layer = self.get_layer(layer_id)
        if layer:
            self.layers.remove(layer)
            # 如果删除的是当前图层，切换到第一个图层
            if self.current_layer_id == layer_id:
                self.current_layer_id = self.layers[0].layer_id
            return True
        return False

    def get_layer(self, layer_id: int) -> Optional[Layer]:
        """获取指定ID的图层"""
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        return None

    def get_current_layer(self) -> Optional[Layer]:
        """获取当前图层"""
        return self.get_layer(self.current_layer_id)

    def set_current_layer(self, layer_id: int):
        """设置当前图层"""
        if self.get_layer(layer_id):
            self.current_layer_id = layer_id

    def move_layer_up(self, layer_id: int) -> bool:
        """向上移动图层"""
        layer = self.get_layer(layer_id)
        if not layer:
            return False

        index = self.layers.index(layer)
        if index < len(self.layers) - 1:
            self.layers[index], self.layers[index + 1] = (
                self.layers[index + 1],
                self.layers[index],
            )
            return True
        return False

    def move_layer_down(self, layer_id: int) -> bool:
        """向下移动图层"""
        layer = self.get_layer(layer_id)
        if not layer:
            return False

        index = self.layers.index(layer)
        if index > 0:
            self.layers[index], self.layers[index - 1] = (
                self.layers[index - 1],
                self.layers[index],
            )
            return True
        return False

    def get_all_instances(self) -> List[MaterialInstance]:
        """获取所有图层的所有实例"""
        instances = []
        for layer in self.layers:
            if layer.visible:
                instances.extend([inst for inst in layer.instances if inst.visible])
        return instances

    def get_instances_at_point(self, x: int, y: int) -> List[MaterialInstance]:
        """获取指定点的所有素材实例（按图层顺序）"""
        instances = []
        # 从上到下遍历图层（倒序）
        for layer in reversed(self.layers):
            if layer.visible:
                layer_instances = layer.get_instances_at_point(x, y)
                instances.extend(layer_instances)
        return instances

    def add_instance_to_current_layer(self, instance: MaterialInstance):
        """添加素材实例到当前图层"""
        current_layer = self.get_current_layer()
        if current_layer:
            current_layer.add_instance(instance)

    def remove_instance(self, instance: MaterialInstance):
        """从所有图层中移除素材实例"""
        for layer in self.layers:
            layer.remove_instance(instance)

    def clear_all_layers(self):
        """清空所有图层"""
        for layer in self.layers:
            layer.clear()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "current_layer_id": self.current_layer_id,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def clear_layers(self):
        """清空所有图层"""
        self.layers.clear()
        self._next_layer_id = 1
        # 重新创建默认图层
        self.add_layer("默认图层")

    def get_layers_in_order(self) -> List[Layer]:
        """按顺序获取所有图层"""
        return self.layers.copy()
