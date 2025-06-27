"""
控制面板组件
包含素材列表、图层管理和属性编辑功能
"""

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.layer import Layer, LayerManager
from core.material import MaterialInfo, MaterialInstance, MaterialManager
from ui.color_picker import BackgroundColorPicker


class MaterialListWidget(QWidget):
    """素材列表组件"""

    material_selected = Signal(str)  # 素材选择信号

    def __init__(self, material_manager: MaterialManager):
        super().__init__()
        self.material_manager = material_manager
        self._init_ui()

    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 标题
        title_label = QLabel("素材库")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)

        # 素材列表
        self.material_list = QListWidget()
        self.material_list.setIconSize(QSize(64, 64))
        self.material_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.material_list)

        # 刷新按钮
        refresh_button = QPushButton("刷新素材")
        refresh_button.clicked.connect(self.refresh_materials)
        layout.addWidget(refresh_button)

    def _on_item_clicked(self, item: QListWidgetItem):
        """素材项点击事件"""
        material_path = item.data(Qt.ItemDataRole.UserRole)
        if material_path:
            self.material_selected.emit(material_path)

    def refresh_materials(self):
        """刷新素材列表"""
        self.material_list.clear()

        for material_info in self.material_manager.materials.values():
            item = QListWidgetItem()
            item.setText(material_info.name)
            item.setData(Qt.ItemDataRole.UserRole, material_info.image_path)

            # 设置缩略图
            if Path(material_info.image_path).exists():
                pixmap = QPixmap(material_info.image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        64,
                        64,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    item.setIcon(QIcon(scaled_pixmap))

            self.material_list.addItem(item)


class LayerTreeWidget(QWidget):
    """图层树组件"""

    layer_visibility_changed = Signal(str, bool)  # 图层可见性改变信号
    layer_selected = Signal(str)  # 图层选择信号
    instance_visibility_changed = Signal(object, bool)  # 素材实例可见性改变信号

    def __init__(self, layer_manager: LayerManager):
        super().__init__()
        self.layer_manager = layer_manager
        self._init_ui()

    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 标题
        title_label = QLabel("图层管理")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)

        # 图层树
        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderLabels(["图层", "可见"])
        self.layer_tree.setRootIsDecorated(False)
        self.layer_tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.layer_tree)

        # 按钮区域
        button_layout = QHBoxLayout()

        # 添加图层按钮
        add_button = QPushButton("添加图层")
        add_button.clicked.connect(self._add_layer)
        button_layout.addWidget(add_button)

        # 删除选中按钮
        delete_button = QPushButton("删除选中")
        delete_button.clicked.connect(self._delete_layer)
        button_layout.addWidget(delete_button)

        layout.addLayout(button_layout)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """图层项点击事件"""
        layer_id = item.data(0, Qt.ItemDataRole.UserRole)

        if column == 1:
            # 点击可见性复选框
            checkbox = self.layer_tree.itemWidget(item, 1)
            if checkbox and isinstance(checkbox, QCheckBox):
                visible = checkbox.isChecked()
                self.layer_visibility_changed.emit(str(layer_id), visible)
        else:
            # 选择图层
            if layer_id:
                self.layer_selected.emit(str(layer_id))

    def _add_layer(self):
        """添加图层"""
        layer = self.layer_manager.add_layer(
            f"图层 {len(self.layer_manager.layers) + 1}"
        )
        self.refresh_layers()

    def _delete_layer(self):
        """删除选中的图层或素材实例"""
        current_item = self.layer_tree.currentItem()
        if current_item:
            # 检查是否是素材实例
            instance = current_item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(instance, MaterialInstance):
                # 删除素材实例
                self.layer_manager.remove_instance(instance)
                self.refresh_layers()
                print(f"已删除素材实例: {instance.material_name}")
            else:
                # 删除整个图层
                layer_id = current_item.data(0, Qt.ItemDataRole.UserRole)
                if layer_id is not None:
                    layer = self.layer_manager.get_layer(layer_id)
                    if layer and len(layer.instances) > 0:
                        # 删除图层中的所有素材实例
                        for instance in layer.instances[
                            :
                        ]:  # 使用切片复制以避免修改时迭代
                            self.layer_manager.remove_instance(instance)
                        self.refresh_layers()
                        print(f"已删除图层 {layer_id} 中的所有素材")

    def _on_instance_visibility_changed(self, instance, visible: bool):
        """素材实例可见性改变"""
        instance.visible = visible
        # 可以发出信号通知画布更新
        # 这里暂时不发信号，由上层处理
        self.instance_visibility_changed.emit(instance, visible)

    def refresh_layers(self):
        """刷新图层列表"""
        self.layer_tree.clear()

        # 阻止信号避免在UI更新时触发额外的信号
        self.layer_tree.blockSignals(True)

        if self.layer_manager:
            for layer in self.layer_manager.layers:
                # 创建图层项
                layer_item = QTreeWidgetItem()
                layer_item.setText(0, f"图层 {layer.layer_id}")
                layer_item.setData(0, Qt.ItemDataRole.UserRole, layer.layer_id)

                # 图层可见性复选框
                layer_checkbox = QCheckBox()
                layer_checkbox.setChecked(layer.visible)
                layer_checkbox.stateChanged.connect(
                    lambda state, lid=layer.layer_id: self._on_layer_visibility_changed(
                        lid, state == Qt.CheckState.Checked.value
                    )
                )
                self.layer_tree.setItemWidget(layer_item, 1, layer_checkbox)

                # 添加素材实例作为子项
                for instance in layer.instances:
                    instance_item = QTreeWidgetItem(layer_item)
                    instance_item.setText(0, instance.material_name)
                    instance_item.setData(0, Qt.ItemDataRole.UserRole, instance)

                    # 素材实例可见性复选框
                    instance_checkbox = QCheckBox()
                    instance_checkbox.setChecked(instance.visible)
                    instance_checkbox.stateChanged.connect(
                        lambda state,
                        inst=instance: self._on_instance_visibility_changed(
                            inst, state == Qt.CheckState.Checked.value
                        )
                    )
                    self.layer_tree.setItemWidget(instance_item, 1, instance_checkbox)

                self.layer_tree.addTopLevelItem(layer_item)
                layer_item.setExpanded(True)  # 默认展开图层

        self.layer_tree.blockSignals(False)

    def select_layer(self, layer_id: int):
        """选中指定图层"""
        for i in range(self.layer_tree.topLevelItemCount()):
            item = self.layer_tree.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole) == layer_id:
                self.layer_tree.setCurrentItem(item)
                break

    def _on_layer_visibility_changed(self, layer_id: int, visible: bool):
        """图层可见性改变处理"""
        self.layer_visibility_changed.emit(str(layer_id), visible)


class PropertyEditor(QWidget):
    """属性编辑器"""

    property_changed = Signal(str, str, object)  # 属性改变信号

    def __init__(self):
        super().__init__()
        self.current_instance = None
        self._init_ui()

        # 启用键盘焦点和事件
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 标题
        title_label = QLabel("属性编辑器")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)

        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll_area)

        # 属性容器
        self.property_container = QWidget()
        self.property_layout = QVBoxLayout(self.property_container)
        scroll_area.setWidget(self.property_container)

        # 初始化背景取色器
        self.color_picker = BackgroundColorPicker()
        self.color_picker.color_picked.connect(self._on_background_color_picked)

        # 默认显示无选择状态
        self._show_no_selection()

    def _show_no_selection(self):
        """显示无选择状态"""
        self._clear_properties()

        no_selection_label = QLabel("未选择对象")
        no_selection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_selection_label.setStyleSheet("color: gray;")
        self.property_layout.addWidget(no_selection_label)

        # 添加快捷键说明
        shortcuts_label = QLabel(
            "快捷键说明:\n"
            "[ ] - 旋转 ±15°\n"
            "Ctrl + / - - 缩放 ±0.1\n"
            "方向键 - 位置微调 ±1px"
        )
        shortcuts_label.setStyleSheet("color: gray; font-size: 10px; margin-top: 10px;")
        shortcuts_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.property_layout.addWidget(shortcuts_label)

    def _clear_properties(self):
        """清空属性控件"""
        while self.property_layout.count():
            child = self.property_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def set_current_instance(self, instance):
        """设置当前选中的素材实例"""
        self.current_instance = instance
        if instance:
            self._update_properties()
            # 确保取色器有背景图像
            if hasattr(self, "_background_image"):
                self.color_picker.set_background_image(self._background_image)
        else:
            self._show_no_selection()

    def _update_properties(self):
        """更新属性显示"""
        self._clear_properties()

        if not self.current_instance:
            self._show_no_selection()
            return

        # 位置属性组
        position_group = QGroupBox("位置")
        position_layout = QVBoxLayout(position_group)

        # X坐标
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_spinbox = QSpinBox()
        self.x_spinbox.setRange(-9999, 9999)
        self.x_spinbox.setValue(int(self.current_instance.x))
        self.x_spinbox.valueChanged.connect(
            lambda value: self.property_changed.emit("position", "x", value)
        )
        x_layout.addWidget(self.x_spinbox)
        position_layout.addLayout(x_layout)

        # Y坐标
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_spinbox = QSpinBox()
        self.y_spinbox.setRange(-9999, 9999)
        self.y_spinbox.setValue(int(self.current_instance.y))
        self.y_spinbox.valueChanged.connect(
            lambda value: self.property_changed.emit("position", "y", value)
        )
        y_layout.addWidget(self.y_spinbox)
        position_layout.addLayout(y_layout)

        self.property_layout.addWidget(position_group)

        # 混合模式组（移到位置和变换之间）
        blend_group = QGroupBox("混合模式")
        blend_layout = QVBoxLayout(blend_group)

        blend_layout.addWidget(QLabel("模式:"))
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["普通", "泊松融合(正常)", "泊松融合(混合)"])

        # 设置当前值
        mode_map = {
            "normal": "普通",
            "poisson_normal": "泊松融合(正常)",
            "poisson_mixed": "泊松融合(混合)",
        }
        current_mode = getattr(self.current_instance, "blend_mode", "normal")
        display_mode = mode_map.get(current_mode, "普通")
        self.blend_combo.setCurrentText(display_mode)

        self.blend_combo.currentTextChanged.connect(self._on_blend_mode_changed)
        blend_layout.addWidget(self.blend_combo)

        self.property_layout.addWidget(blend_group)

        # 变换属性组
        transform_group = QGroupBox("变换")
        transform_layout = QVBoxLayout(transform_group)

        # 缩放
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("缩放:"))
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.1, 10.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setDecimals(2)
        self.scale_spinbox.setValue(self.current_instance.scale)
        self.scale_spinbox.valueChanged.connect(
            lambda value: self.property_changed.emit("transform", "scale", value)
        )
        scale_layout.addWidget(self.scale_spinbox)
        transform_layout.addLayout(scale_layout)

        # 旋转
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("旋转:"))
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(0, 359)
        self.rotation_slider.setValue(int(self.current_instance.rotation))
        self.rotation_spinbox = QSpinBox()
        self.rotation_spinbox.setRange(0, 359)
        self.rotation_spinbox.setValue(int(self.current_instance.rotation))

        # 连接滑块和数值框
        self.rotation_slider.valueChanged.connect(self.rotation_spinbox.setValue)
        self.rotation_spinbox.valueChanged.connect(self.rotation_slider.setValue)
        self.rotation_slider.valueChanged.connect(
            lambda value: self.property_changed.emit("transform", "rotation", value)
        )

        rotation_layout.addWidget(self.rotation_slider)
        rotation_layout.addWidget(self.rotation_spinbox)
        transform_layout.addLayout(rotation_layout)

        self.property_layout.addWidget(transform_group)

        # 色彩叠加控件组
        color_group = QGroupBox("色彩叠加")
        color_layout = QFormLayout(color_group)

        # 颜色选择行
        color_row_layout = QHBoxLayout()

        # 颜色按钮
        self.color_button = QPushButton("透明")
        self.color_button.setMinimumSize(50, 30)
        self.color_button.setStyleSheet(
            "background-color: transparent; border: 1px solid gray;"
        )
        self.color_button.clicked.connect(self._on_color_button_clicked)
        color_row_layout.addWidget(self.color_button)

        # 清除颜色按钮
        clear_color_btn = QPushButton("清除")
        clear_color_btn.setMaximumSize(50, 30)
        clear_color_btn.clicked.connect(self._on_clear_color_clicked)
        color_row_layout.addWidget(clear_color_btn)

        color_row_layout.addStretch()
        color_layout.addRow("颜色:", color_row_layout)

        # 透明度控件
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(0)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)

        self.opacity_spinbox = QSpinBox()
        self.opacity_spinbox.setRange(0, 100)
        self.opacity_spinbox.setSuffix("%")
        self.opacity_spinbox.setValue(0)
        self.opacity_spinbox.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_spinbox)

        color_layout.addRow("透明度:", opacity_layout)

        self.property_layout.addWidget(color_group)

        # 背景取色器
        self.color_picker = BackgroundColorPicker()
        self.color_picker.color_picked.connect(self._on_background_color_picked)
        self.property_layout.addWidget(self.color_picker)

    def _on_blend_mode_changed(self, text: str):
        """混合模式改变事件"""
        if not hasattr(self, "current_instance") or not self.current_instance:
            return

        # 转换显示文本到内部值
        mode_map = {
            "普通": "normal",
            "泊松融合(正常)": "poisson_normal",
            "泊松融合(混合)": "poisson_mixed",
        }

        mode = mode_map.get(text, "normal")
        self.current_instance.blend_mode = mode
        self.property_changed.emit("blend", "mode", mode)

    def _update_values_only(self):
        """只更新数值显示，不重新创建控件"""
        if not self.current_instance:
            return

        # 更新位置控件的值
        if hasattr(self, "x_spinbox"):
            self.x_spinbox.blockSignals(True)
            self.x_spinbox.setValue(int(self.current_instance.x))
            self.x_spinbox.blockSignals(False)

        if hasattr(self, "y_spinbox"):
            self.y_spinbox.blockSignals(True)
            self.y_spinbox.setValue(int(self.current_instance.y))
            self.y_spinbox.blockSignals(False)

        # 更新缩放控件的值
        if hasattr(self, "scale_spinbox"):
            self.scale_spinbox.blockSignals(True)
            self.scale_spinbox.setValue(self.current_instance.scale)
            self.scale_spinbox.blockSignals(False)

        # 更新旋转控件的值
        if hasattr(self, "rotation_slider"):
            self.rotation_slider.blockSignals(True)
            self.rotation_slider.setValue(int(self.current_instance.rotation))
            self.rotation_slider.blockSignals(False)

        if hasattr(self, "rotation_spinbox"):
            self.rotation_spinbox.blockSignals(True)
            self.rotation_spinbox.setValue(int(self.current_instance.rotation))
            self.rotation_spinbox.blockSignals(False)

        # 更新混合模式控件的值
        if hasattr(self, "blend_combo"):
            mode_map = {
                "normal": "普通",
                "poisson_normal": "泊松融合(正常)",
                "poisson_mixed": "泊松融合(混合)",
            }
            current_mode = getattr(self.current_instance, "blend_mode", "normal")
            display_mode = mode_map.get(current_mode, "普通")
            self.blend_combo.blockSignals(True)
            self.blend_combo.setCurrentText(display_mode)
            self.blend_combo.blockSignals(False)

        # 更新色彩叠加控件的值
        if hasattr(self, "color_button"):
            current_color = getattr(self.current_instance, "color_overlay", None)
            if current_color:
                color = QColor(current_color[0], current_color[1], current_color[2])
                self.color_button.setStyleSheet(
                    f"background-color: {color.name()}; border: 1px solid gray;"
                )
                self.color_button.setText("")
            else:
                self.color_button.setStyleSheet(
                    "background-color: transparent; border: 1px solid gray;"
                )
                self.color_button.setText("透明")

        if hasattr(self, "opacity_slider"):
            current_opacity = getattr(self.current_instance, "overlay_opacity", 0.0)
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(int(current_opacity * 100))
            self.opacity_slider.blockSignals(False)

        if hasattr(self, "opacity_spinbox"):
            current_opacity = getattr(self.current_instance, "overlay_opacity", 0.0)
            self.opacity_spinbox.blockSignals(True)
            self.opacity_spinbox.setValue(int(current_opacity * 100))
            self.opacity_spinbox.blockSignals(False)

    def _on_color_button_clicked(self):
        """颜色按钮点击事件"""
        color_dialog = QColorDialog()
        color = color_dialog.getColor()
        if color.isValid():
            self.current_instance.color_overlay = (
                color.red(),
                color.green(),
                color.blue(),
            )
            self.property_changed.emit(
                "color_overlay", "color", self.current_instance.color_overlay
            )
            self._update_values_only()

    def _on_clear_color_clicked(self):
        """清除颜色按钮点击事件"""
        self.current_instance.color_overlay = None
        self.property_changed.emit(
            "color_overlay", "color", self.current_instance.color_overlay
        )
        self._update_values_only()

    def _on_opacity_changed(self, value: int):
        """透明度改变事件"""
        if not hasattr(self, "current_instance") or not self.current_instance:
            return

        opacity = value / 100.0
        self.current_instance.overlay_opacity = opacity
        self.property_changed.emit("overlay_opacity", "opacity", opacity)
        self._update_values_only()

    def _on_background_color_picked(self, color: tuple):
        """背景颜色被选中事件"""
        if not hasattr(self, "current_instance") or not self.current_instance:
            return

        self.current_instance.color_overlay = color
        self.property_changed.emit(
            "color_overlay", "color", self.current_instance.color_overlay
        )
        self._update_values_only()

    def _set_background_image_cache(self, image):
        """缓存背景图像"""
        self._background_image = image
        self.set_background_image(image)

    def set_background_image(self, image):
        """设置背景图像"""
        self._background_image = image
        if hasattr(self, "color_picker") and self.color_picker:
            print(f"设置取色器背景图像: {image is not None}")
            self.color_picker.set_background_image(image)
        else:
            print("取色器未初始化")


class ControlPanel(QWidget):
    """控制面板主组件"""

    material_selected = Signal(str)
    layer_visibility_changed = Signal(str, bool)
    layer_selected = Signal(str)
    property_changed = Signal(str, str, object)
    instance_visibility_changed = Signal(object, bool)  # 素材实例可见性改变信号

    def __init__(self, material_manager: MaterialManager, layer_manager: LayerManager):
        super().__init__()
        self.material_manager = material_manager
        self.layer_manager = layer_manager
        self._init_ui()

    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建标签页
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # 素材标签页
        self.material_widget = MaterialListWidget(self.material_manager)
        self.material_widget.material_selected.connect(self.material_selected)
        self.tab_widget.addTab(self.material_widget, "素材")

        # 图层标签页
        self.layer_widget = LayerTreeWidget(self.layer_manager)
        self.layer_widget.layer_visibility_changed.connect(
            self.layer_visibility_changed
        )
        self.layer_widget.layer_selected.connect(self.layer_selected)
        self.layer_widget.instance_visibility_changed.connect(
            self.instance_visibility_changed
        )
        self.tab_widget.addTab(self.layer_widget, "图层")

        # 属性标签页
        self.property_widget = PropertyEditor()
        self.property_widget.property_changed.connect(self.property_changed)
        self.tab_widget.addTab(self.property_widget, "属性")

    def refresh_materials(self):
        """刷新素材列表"""
        self.material_widget.refresh_materials()

    def refresh_layers(self):
        """刷新图层列表"""
        self.layer_widget.refresh_layers()

    def set_current_instance(self, instance):
        """设置当前选中的素材实例"""
        self.property_widget.set_current_instance(instance)

    def set_active_tab(self, tab_name: str):
        """设置活动标签页"""
        if tab_name == "素材":
            self.tab_widget.setCurrentIndex(0)
        elif tab_name == "图层":
            self.tab_widget.setCurrentIndex(1)
        elif tab_name == "属性":
            self.tab_widget.setCurrentIndex(2)

    def select_layer(self, layer_id: int):
        """选中指定图层"""
        self.layer_widget.select_layer(layer_id)

    def set_background_image(self, image):
        """设置背景图像"""
        if hasattr(self, "color_picker") and self.color_picker:
            print(f"设置取色器背景图像: {image is not None}")
            self.color_picker.set_background_image(image)
        else:
            print("取色器未初始化")
