#!/usr/bin/env python3
"""
素材编辑器主程序
基于PyQt6和OpenCV的图像合成工具
"""

import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.export import ProjectExporter
from core.layer import LayerManager
from core.material import MaterialManager
from ui.controls import ControlPanel
from ui.dialogs import RandomGenerateDialog
from ui.graphics_canvas import HighPerformanceCanvasWidget
from utils.file_utils import FileManager


class MaterialEditor(QMainWindow):
    """素材编辑器主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("素材编辑器")
        self.setGeometry(100, 100, 1200, 900)

        # 核心组件
        self.material_manager = MaterialManager()
        self.layer_manager = LayerManager()
        self.file_manager = FileManager()
        self.project_exporter = ProjectExporter()

        # 当前项目状态
        self.current_background_path: Optional[str] = None
        self.is_modified = False
        self.current_selected_material: Optional[str] = None  # 当前选择的素材路径

        # 背景图像管理
        self.background_images: List[str] = []  # 背景图像路径列表
        self.current_background_index: int = 0  # 当前背景图像索引

        # 添加模式：'click' 或 'double_click'
        self.add_mode = "double_click"

        # 初始化界面
        self._init_ui()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()
        self._connect_signals()

        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(100)  # 100ms更新一次

    def _init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧控制面板
        self.control_panel = ControlPanel(self.material_manager, self.layer_manager)
        self.control_panel.setMinimumWidth(300)
        self.control_panel.setMaximumWidth(400)
        splitter.addWidget(self.control_panel)

        # 右侧画布区域 - 使用新的高性能画布
        self.canvas_area = HighPerformanceCanvasWidget()
        self.canvas_area.set_layer_manager(self.layer_manager)
        splitter.addWidget(self.canvas_area)

        # 设置分割器比例
        splitter.setSizes([300, 1000])

    def _init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        # 加载背景目录
        load_bg_action = QAction("加载背景目录(&B)", self)
        load_bg_action.triggered.connect(self.load_background_directory)
        file_menu.addAction(load_bg_action)

        # 加载素材目录
        load_materials_action = QAction("加载素材目录(&M)", self)
        load_materials_action.triggered.connect(self.load_materials)
        file_menu.addAction(load_materials_action)

        file_menu.addSeparator()

        # 清空所有素材
        clear_materials_action = QAction("清空所有素材(&C)", self)
        clear_materials_action.triggered.connect(self.clear_all_materials)
        file_menu.addAction(clear_materials_action)

        file_menu.addSeparator()

        # 一键导出
        export_all_action = QAction("一键导出图像和标注(&E)", self)
        export_all_action.triggered.connect(self.export_all)
        file_menu.addAction(export_all_action)

        file_menu.addSeparator()

        # 退出
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = menubar.addMenu("编辑(&E)")

        # 删除选中
        delete_action = QAction("删除选中(&D)", self)
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_action)

        edit_menu.addSeparator()

        # 随机生成物体
        random_generate_action = QAction("随机生成物体(&R)", self)
        random_generate_action.setShortcut(QKeySequence("Ctrl+R"))
        random_generate_action.triggered.connect(self.show_random_generate_dialog)
        edit_menu.addAction(random_generate_action)

        edit_menu.addSeparator()

        # 添加模式子菜单
        add_mode_menu = edit_menu.addMenu("添加模式(&A)")

        # 单击添加
        click_add_action = QAction("单击添加", self)
        click_add_action.setCheckable(True)
        click_add_action.setChecked(self.add_mode == "click")
        click_add_action.triggered.connect(lambda: self.set_add_mode("click"))
        add_mode_menu.addAction(click_add_action)

        # 双击添加
        double_click_add_action = QAction("双击添加", self)
        double_click_add_action.setCheckable(True)
        double_click_add_action.setChecked(self.add_mode == "double_click")
        double_click_add_action.triggered.connect(
            lambda: self.set_add_mode("double_click")
        )
        add_mode_menu.addAction(double_click_add_action)

        # 保存菜单动作引用
        self.click_add_action = click_add_action
        self.double_click_add_action = double_click_add_action

        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")

        # 适应窗口
        fit_action = QAction("适应窗口(&F)", self)
        fit_action.triggered.connect(self.canvas_area.zoom_to_fit)
        view_menu.addAction(fit_action)

        # 实际大小
        actual_size_action = QAction("实际大小(&A)", self)
        actual_size_action.triggered.connect(self.canvas_area.zoom_to_actual_size)
        view_menu.addAction(actual_size_action)

        view_menu.addSeparator()

        # 性能设置子菜单
        performance_menu = view_menu.addMenu("性能设置(&P)")

        # 增量更新选项
        self.incremental_update_action = QAction("启用增量更新", self)
        self.incremental_update_action.setCheckable(True)
        self.incremental_update_action.setChecked(True)
        self.incremental_update_action.setToolTip(
            "拖动时只重新计算当前素材，提高多素材时的性能"
        )
        self.incremental_update_action.triggered.connect(
            self._toggle_incremental_update
        )
        performance_menu.addAction(self.incremental_update_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")

        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _init_toolbar(self):
        """初始化工具栏"""
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)

        # 加载背景目录
        load_bg_action = QAction("加载背景目录", self)
        load_bg_action.triggered.connect(self.load_background_directory)
        toolbar.addAction(load_bg_action)

        # 上一张背景
        self.prev_bg_action = QAction("上一张背景", self)
        self.prev_bg_action.triggered.connect(self.prev_background)
        self.prev_bg_action.setEnabled(False)
        toolbar.addAction(self.prev_bg_action)

        # 下一张背景
        self.next_bg_action = QAction("下一张背景", self)
        self.next_bg_action.triggered.connect(self.next_background)
        self.next_bg_action.setEnabled(False)
        toolbar.addAction(self.next_bg_action)

        toolbar.addSeparator()

        # 加载素材
        load_materials_action = QAction("加载素材", self)
        load_materials_action.triggered.connect(self.load_materials)
        toolbar.addAction(load_materials_action)

        # 清空素材
        clear_materials_action = QAction("清空素材", self)
        clear_materials_action.triggered.connect(self.clear_all_materials)
        toolbar.addAction(clear_materials_action)

        # 随机生成
        random_generate_action = QAction("随机生成", self)
        random_generate_action.triggered.connect(self.show_random_generate_dialog)
        toolbar.addAction(random_generate_action)

        toolbar.addSeparator()

        # 一键导出
        export_all_action = QAction("一键导出", self)
        export_all_action.triggered.connect(self.export_all)
        toolbar.addAction(export_all_action)

        toolbar.addSeparator()

        # 适应窗口
        fit_action = QAction("适应窗口", self)
        fit_action.triggered.connect(self.canvas_area.zoom_to_fit)
        toolbar.addAction(fit_action)

    def _init_statusbar(self):
        """初始化状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 状态标签
        self.status_label = QLabel(
            "就绪 - 左键点击/拖动素材，右键拖动画布，滚轮以鼠标为中心缩放"
        )
        self.status_bar.addWidget(self.status_label)

        # 坐标标签
        self.coord_label = QLabel("坐标: (0, 0)")
        self.status_bar.addPermanentWidget(self.coord_label)

        # 缩放标签
        self.zoom_label = QLabel("缩放: 100%")
        self.status_bar.addPermanentWidget(self.zoom_label)

        # 添加模式标签
        self.add_mode_label = QLabel(
            f"添加模式: {'双击' if self.add_mode == 'double_click' else '单击'}"
        )
        self.status_bar.addPermanentWidget(self.add_mode_label)

    def _connect_signals(self):
        """连接信号槽"""
        # 画布信号
        self.canvas_area.instance_selected.connect(self._on_selection_changed)
        self.canvas_area.instance_moved.connect(self._on_canvas_modified)
        self.canvas_area.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas_area.canvas_double_clicked.connect(self._on_canvas_double_clicked)

        # 控制面板信号
        self.control_panel.material_selected.connect(self._on_material_selected)
        self.control_panel.layer_visibility_changed.connect(
            self._on_layer_visibility_changed
        )
        self.control_panel.property_changed.connect(self._on_property_changed)
        self.control_panel.instance_visibility_changed.connect(
            self._on_instance_visibility_changed
        )

    def set_add_mode(self, mode: str):
        """设置添加模式"""
        self.add_mode = mode

        # 更新菜单状态
        self.click_add_action.setChecked(mode == "click")
        self.double_click_add_action.setChecked(mode == "double_click")

        # 更新状态栏
        self.add_mode_label.setText(
            f"添加模式: {'双击' if mode == 'double_click' else '单击'}"
        )

    def _on_canvas_clicked(self, x: int, y: int):
        """画布单击事件"""
        # 检查是否正在使用取色器
        if hasattr(self.control_panel.property_widget, "color_picker"):
            color_picker = self.control_panel.property_widget.color_picker
            if color_picker.is_sampling_enabled():
                # 进行取色操作
                color_picker.sample_color_at_position(x, y)
                return

        if self.add_mode == "single_click":
            self._add_material_at_position(x, y)

    def _on_canvas_double_clicked(self, x: int, y: int):
        """画布双击事件"""
        if self.add_mode == "double_click":
            self._add_material_at_position(x, y)

    def _add_material_at_position(self, x: int, y: int):
        """在指定位置添加素材"""
        if self.current_selected_material:
            # 如果有选择的素材，在点击位置添加素材
            try:
                # 从素材路径获取素材名称
                material_name = Path(self.current_selected_material).stem
                print(f"添加素材: {material_name} 到位置 ({x}, {y})")

                # 创建素材实例
                instance = self.material_manager.create_instance(
                    material_name,
                    x,
                    y,
                    scale=1.0,
                    rotation=0.0,
                    layer_id=self.layer_manager.current_layer_id,
                )

                if instance:
                    # 添加到当前图层
                    self.layer_manager.add_instance_to_current_layer(instance)

                    # 检查图层中的实例数量
                    current_layer = self.layer_manager.get_current_layer()
                    if current_layer:
                        print(f"当前图层实例数量: {len(current_layer.instances)}")

                    # 更新画布
                    self.canvas_area.update_canvas()

                    # 刷新图层面板显示
                    self.control_panel.refresh_layers()

                    # 设置为选中状态
                    self.canvas_area.selected_instance = instance

                    # 标记项目已修改
                    self.is_modified = True
                    self._update_window_title()

                    # 更新状态栏
                    self.status_label.setText(f"已添加素材: {material_name}")

                    # 清除当前选择的素材（可选，如果希望连续添加则注释掉这行）
                    # self.current_selected_material = None
                else:
                    self.status_label.setText("添加素材失败")
                    print("素材实例创建失败")

            except Exception as e:
                print(f"添加素材异常: {e}")
                import traceback

                traceback.print_exc()
                QMessageBox.critical(self, "错误", f"添加素材失败: {str(e)}")
                self.status_label.setText("添加素材失败")
        else:
            # 没有选择素材时，显示点击坐标
            self.status_label.setText(f"点击位置: ({x}, {y})")

    def _on_mouse_moved(self, x: int, y: int):
        """鼠标移动事件"""
        self.coord_label.setText(f"坐标: ({x}, {y})")

    def _on_selection_changed(self):
        """选择改变事件"""
        try:
            # 获取当前选择的实例
            selected_item = None
            selected_instance = None

            # 从graphics_view获取选择的项目
            if hasattr(self.canvas_area, "graphics_view"):
                selected_items = self.canvas_area.graphics_view.scene.selectedItems()
                if selected_items:
                    selected_item = selected_items[0]
                    if hasattr(selected_item, "instance"):
                        selected_instance = selected_item.instance

            # 更新控制面板
            self.control_panel.set_current_instance(selected_instance)

            # 更新状态栏
            if selected_instance:
                self.status_label.setText(
                    f"已选择素材: {selected_instance.material_info.name} "
                    f"位置: ({selected_instance.x}, {selected_instance.y}) "
                    f"缩放: {selected_instance.scale:.2f} "
                    f"旋转: {selected_instance.rotation:.1f}°"
                )
            else:
                self.status_label.setText("未选择素材")

        except Exception as e:
            print(f"选择改变事件处理失败: {e}")
            import traceback

            traceback.print_exc()

    def _on_canvas_modified(self):
        """画布修改事件"""
        self.is_modified = True
        self._update_window_title()

    def _on_material_selected(self, material_path: str):
        """素材选择事件"""
        # 设置当前选择的素材，等待用户点击画布放置
        self.current_selected_material = material_path
        material_name = Path(material_path).stem
        mode_text = "双击" if self.add_mode == "double_click" else "单击"
        self.status_label.setText(
            f"已选择素材: {material_name}，{mode_text}画布空白区域放置素材"
        )

    def _on_layer_visibility_changed(self, layer_id: str, visible: bool):
        """图层可见性改变事件"""
        try:
            if layer_id and layer_id.strip():  # 检查layer_id是否为空
                layer = self.layer_manager.get_layer(int(layer_id))
                if layer:
                    layer.visible = visible
                    self.canvas_area.update_canvas()
        except (ValueError, TypeError) as e:
            print(f"图层可见性改变失败: {e}, layer_id='{layer_id}'")

    def _on_property_changed(self, group: str, property_name: str, value):
        """素材属性改变事件"""
        try:
            # 获取当前选择的实例
            selected_instance = None
            if hasattr(self.canvas_area, "graphics_view"):
                selected_items = self.canvas_area.graphics_view.scene.selectedItems()
                if selected_items and hasattr(selected_items[0], "instance"):
                    selected_instance = selected_items[0].instance

            if selected_instance:
                # 更新素材实例属性
                if group == "position":
                    if property_name == "x":
                        selected_instance.x = value
                        # 更新图形项位置
                        if hasattr(self.canvas_area, "graphics_view"):
                            for (
                                instance,
                                item,
                            ) in self.canvas_area.graphics_view.material_items.items():
                                if instance == selected_instance:
                                    item.setPos(value, selected_instance.y)
                                    break
                    elif property_name == "y":
                        selected_instance.y = value
                        # 更新图形项位置
                        if hasattr(self.canvas_area, "graphics_view"):
                            for (
                                instance,
                                item,
                            ) in self.canvas_area.graphics_view.material_items.items():
                                if instance == selected_instance:
                                    item.setPos(selected_instance.x, value)
                                    break
                elif group == "transform":
                    if property_name == "scale":
                        selected_instance.scale = value
                        # 清除变换缓存以重新计算
                        selected_instance._transformed_image_cache = None
                        selected_instance._transformed_mask_cache = None
                        selected_instance._transform_cache_key = None
                    elif property_name == "rotation":
                        selected_instance.rotation = value
                        # 清除变换缓存以重新计算
                        selected_instance._transformed_image_cache = None
                        selected_instance._transformed_mask_cache = None
                        selected_instance._transform_cache_key = None
                elif group == "blend":
                    if property_name == "mode":
                        selected_instance.blend_mode = value
                elif group == "color_overlay":
                    if property_name == "color":
                        selected_instance.color_overlay = value
                elif group == "overlay_opacity":
                    if property_name == "opacity":
                        selected_instance.overlay_opacity = value

                # 使用新的实时更新方法
                if hasattr(self.canvas_area, "update_instance_display"):
                    # 使用高性能组件的实时更新
                    self.canvas_area.update_instance_display(selected_instance)
                else:
                    # 回退到完整更新
                    self.canvas_area.update_canvas(force_full_update=True)

                # 标记项目已修改
                self.is_modified = True
                self._update_window_title()
                self.status_label.setText(f"已更新属性: {property_name} = {value}")
            else:
                print(f"未选择素材实例，无法更新属性: {property_name} = {value}")
        except Exception as e:
            print(f"更新素材属性失败: {e}")
            import traceback
            traceback.print_exc()

    def _update_status(self):
        """更新状态栏"""
        # 更新缩放比例
        if hasattr(self.canvas_area, "graphics_view"):
            # 获取QGraphicsView的变换矩阵
            transform = self.canvas_area.graphics_view.transform()
            scale = transform.m11()  # 水平缩放因子
            self.zoom_label.setText(f"缩放: {scale * 100:.0f}%")
        else:
            self.zoom_label.setText("缩放: 100%")

    def _update_window_title(self):
        """更新窗口标题"""
        title = "素材编辑器"
        if self.current_background_path:
            filename = Path(self.current_background_path).name
            if len(self.background_images) > 1:
                # 显示当前图像索引
                current_index = self.current_background_index + 1
                total_count = len(self.background_images)
                title += f" - {filename} ({current_index}/{total_count})"
            else:
                title += f" - {filename}"
        if self.is_modified:
            title += " *"
        self.setWindowTitle(title)

    def load_background(self):
        """加载单张背景图像（兼容性方法）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            try:
                import cv2
                import numpy as np

                # 加载图像
                image = cv2.imdecode(
                    np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR
                )

                if image is not None:
                    self.canvas_area.set_background_image(image)

                    # 设置取色器的背景图像
                    self.control_panel.property_widget.set_background_image(image)

                    # 更新画布显示
                    self.canvas_area.update_canvas()
                    print(f"背景图像加载成功: {file_path}")
                else:
                    QMessageBox.warning(self, "错误", "无法加载图像文件")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载背景图像失败：{str(e)}")
                print(f"加载背景图像失败: {e}")
                import traceback
                traceback.print_exc()

    def load_materials(self):
        """加载素材目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择素材目录")

        if dir_path:
            try:
                self.material_manager.load_materials_from_directory(dir_path)
                self.control_panel.refresh_materials()
                self.status_label.setText(
                    f"加载了 {len(self.material_manager.materials)} 个素材"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载素材失败: {str(e)}")

    def export_all(self):
        """一键导出图像和标注"""
        if self.canvas_area.background_image is None:
            QMessageBox.warning(self, "警告", "请先加载背景图像")
            return

        if not self.layer_manager or not self.layer_manager.get_all_instances():
            QMessageBox.warning(self, "警告", "没有素材实例可导出")
            return

        # 让用户选择导出目录
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")

        if not export_dir:
            return

        try:
            import os
            from pathlib import Path

            # 获取源文件名（不含扩展名）
            if self.current_background_path:
                source_filename = Path(self.current_background_path).stem
            else:
                source_filename = "image"

            # 生成唯一的文件名
            def get_unique_filename(
                base_dir: str, base_name: str, extension: str
            ) -> Tuple[str, str]:
                """生成唯一的文件名"""
                index = 1
                while True:
                    filename = f"FAKE_{base_name}_{index}"
                    image_path = os.path.join(base_dir, f"{filename}.{extension}")
                    json_path = os.path.join(base_dir, f"{filename}.json")

                    if not os.path.exists(image_path) and not os.path.exists(json_path):
                        return image_path, json_path
                    index += 1

            # 生成唯一的文件路径
            image_path, annotation_path = get_unique_filename(
                export_dir, source_filename, "jpg"
            )

            # 导出图像
            import cv2

            canvas_image = self.canvas_area.background_image.copy()

            # 绘制所有可见的素材实例
            instances = self.layer_manager.get_all_instances()
            for instance in instances:
                if instance.visible:
                    canvas_image = self.canvas_area._draw_instance_on_canvas(
                        canvas_image, instance
                    )

            # 保存为JPG格式，设置质量参数
            success = cv2.imwrite(
                image_path, canvas_image, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if not success:
                QMessageBox.critical(self, "错误", "导出图像失败")
                return

            # 导出标注
            import json

            # 构建标注数据 - 使用AnyMark格式
            annotations = {
                "info": {
                    "description": "AnyMark",
                    "folder": export_dir,
                    "name": Path(image_path).name,
                    "width": self.canvas_area.background_image.shape[1],
                    "height": self.canvas_area.background_image.shape[0],
                    "depth": 3,
                    "note": "",
                },
                "object": [],
            }

            # 添加所有素材实例的标注信息
            instances = self.layer_manager.get_all_instances()
            group_id = 1
            for i, instance in enumerate(instances):
                if instance.visible:
                    # 使用mask边界而不是图像边界
                    x1, y1, x2, y2 = instance.get_mask_bounding_rect()

                    # 计算边界框的中心点和尺寸
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2
                    center_y = y1 + height / 2

                    # 获取mask轮廓的所有点坐标
                    segment_points = instance.get_segment_points()

                    # 转换为所需格式
                    segment = []
                    for point in segment_points:
                        segment.append({"x": float(point["x"]), "y": float(point["y"])})

                    instance_data = {
                        "category": instance.material_name,
                        "category_id": 1,  # 默认分类ID
                        "group": group_id,
                        "segment": segment,
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "w": float(width),
                            "h": float(height),
                            "center_x": float(center_x),
                            "center_y": float(center_y),
                        },
                        "layer": instance.layer_id if instance.layer_id else 1,
                        "note": "",
                    }
                    annotations["object"].append(instance_data)
                    group_id += 1

            # 保存JSON文件
            with open(annotation_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)

            # 获取文件名用于显示
            image_filename = Path(image_path).name
            annotation_filename = Path(annotation_path).name

            self.status_label.setText(
                f"导出成功：图像({image_filename})、标注({annotation_filename})"
            )
            QMessageBox.information(
                self,
                "导出成功",
                f"已成功导出：\n图像文件：{image_path}\n标注文件：{annotation_path}",
            )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于素材编辑器",
            "素材编辑器 v1.0\n\n"
            "基于PyQt6和OpenCV的图像合成工具\n"
            "支持素材的拖拽、缩放、旋转等操作\n"
            "具有完整的图层管理和项目保存功能",
        )

    def delete_selected(self):
        """删除选中的素材实例"""
        selected_instance = self.canvas_area.selected_instance
        if selected_instance:
            # 从图层管理器中删除实例
            self.layer_manager.remove_instance(selected_instance)
            # 清除画布选择
            self.canvas_area.selected_instance = None
            # 更新画布显示
            self.canvas_area.update_canvas()
            # 刷新图层面板
            self.control_panel.refresh_layers()
            # 更新属性编辑器
            self.control_panel.set_current_instance(None)
            # 标记为已修改
            self.is_modified = True
            self._update_window_title()
            self.status_label.setText("已删除选中素材")
        else:
            self.status_label.setText("没有选中的素材可删除")

    def closeEvent(self, event):
        """窗口关闭事件"""
        event.accept()

    def _on_instance_visibility_changed(self, instance, visible: bool):
        """素材实例可见性改变事件"""
        try:
            # 实例的可见性已经在图层树组件中更新了
            # 这里只需要更新画布显示
            self.canvas_area.update_canvas()
            self.is_modified = True
            self._update_window_title()
        except Exception as e:
            print(f"素材实例可见性改变失败: {e}")

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # Delete键删除选中的素材实例
        if event.key() == Qt.Key.Key_Delete:
            selected_instance = self.canvas_area.selected_instance
            if selected_instance:
                # 从图层管理器中删除实例
                self.layer_manager.remove_instance(selected_instance)
                # 清除画布选择
                self.canvas_area.selected_instance = None
                # 更新画布显示
                self.canvas_area.update_canvas()
                # 刷新图层面板
                self.control_panel.refresh_layers()
                # 更新属性编辑器
                self.control_panel.set_current_instance(None)
                # 标记为已修改
                self.is_modified = True
                self._update_window_title()
                self.status_label.setText("已删除选中素材")
                event.accept()
                return
            else:
                self.status_label.setText("没有选中的素材可删除")
                event.accept()
                return

        # 全局快捷键处理：只有在选中素材实例时才响应
        selected_instance = self.canvas_area.selected_instance
        if selected_instance:
            key = event.key()
            modifiers = event.modifiers()
            changed = False

            # Ctrl + 数字键的缩放
            if modifiers == Qt.KeyboardModifier.ControlModifier:
                if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
                    # Ctrl + = 放大
                    new_scale = min(10.0, selected_instance.scale + 0.1)
                    selected_instance.scale = new_scale
                    self._notify_property_changed("transform", "scale", new_scale)
                    changed = True
                elif key == Qt.Key.Key_Minus:
                    # Ctrl + - 缩小
                    new_scale = max(0.1, selected_instance.scale - 0.1)
                    selected_instance.scale = new_scale
                    self._notify_property_changed("transform", "scale", new_scale)
                    changed = True
            else:
                # 普通旋转快捷键
                if key == Qt.Key.Key_BracketLeft:
                    # [ 逆时针旋转
                    new_rotation = (selected_instance.rotation - 15) % 360
                    selected_instance.rotation = new_rotation
                    self._notify_property_changed("transform", "rotation", new_rotation)
                    changed = True
                elif key == Qt.Key.Key_BracketRight:
                    # ] 顺时针旋转
                    new_rotation = (selected_instance.rotation + 15) % 360
                    selected_instance.rotation = new_rotation
                    self._notify_property_changed("transform", "rotation", new_rotation)
                    changed = True
                # 位置微调
                elif key == Qt.Key.Key_Left:
                    new_x = selected_instance.x - 1
                    selected_instance.x = new_x
                    self._notify_property_changed("position", "x", new_x)
                    changed = True
                elif key == Qt.Key.Key_Right:
                    new_x = selected_instance.x + 1
                    selected_instance.x = new_x
                    self._notify_property_changed("position", "x", new_x)
                    changed = True
                elif key == Qt.Key.Key_Up:
                    new_y = selected_instance.y - 1
                    selected_instance.y = new_y
                    self._notify_property_changed("position", "y", new_y)
                    changed = True
                elif key == Qt.Key.Key_Down:
                    new_y = selected_instance.y + 1
                    selected_instance.y = new_y
                    self._notify_property_changed("position", "y", new_y)
                    changed = True

            if changed:
                # 更新画布显示
                self.canvas_area.update_canvas()
                # 更新属性编辑器显示
                self.control_panel.property_widget._update_values_only()
                # 标记为已修改
                self.is_modified = True
                self._update_window_title()
                event.accept()
                return

        # 如果没有处理，传递给父类
        super().keyPressEvent(event)

    def _notify_property_changed(self, group: str, property_name: str, value):
        """通知属性改变（内部调用）"""
        # 清除变换缓存
        if group == "transform":
            selected_instance = self.canvas_area.selected_instance
            if selected_instance:
                selected_instance._transformed_image_cache = None
                selected_instance._transformed_mask_cache = None
                selected_instance._transform_cache_key = None

        # 更新画布显示
        self.canvas_area.update_canvas(force_full_update=True)

        # 更新属性编辑器显示
        self.control_panel.set_current_instance(self.canvas_area.selected_instance)

    def export_image(self):
        """导出图像（兼容性方法）"""
        if self.canvas_area.background_image is None:
            QMessageBox.warning(self, "警告", "请先加载背景图像")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出合成图像",
            "",
            "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*.*)",
        )

        if file_path:
            try:
                import cv2

                # 获取当前的合成图像
                canvas_image = self.canvas_area.background_image.copy()

                # 绘制所有可见的素材实例
                if self.layer_manager:
                    instances = self.layer_manager.get_all_instances()
                    for instance in instances:
                        if instance.visible:
                            canvas_image = self.canvas_area._draw_instance_on_canvas(
                                canvas_image, instance
                            )

                # 保存图像
                success = cv2.imwrite(file_path, canvas_image)

                if success:
                    self.status_label.setText(f"图像已导出到: {file_path}")
                else:
                    QMessageBox.critical(self, "错误", "导出图像失败")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出图像失败: {str(e)}")
                import traceback
                traceback.print_exc()

    def export_annotation(self):
        """导出标注（兼容性方法）"""
        if not self.layer_manager or not self.layer_manager.get_all_instances():
            QMessageBox.warning(self, "警告", "没有素材实例可导出")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出标注文件", "", "JSON文件 (*.json)"
        )

        if file_path:
            try:
                import json

                # 构建标注数据
                annotations = {
                    "image_info": {
                        "width": self.canvas_area.background_image.shape[1]
                        if self.canvas_area.background_image is not None
                        else 0,
                        "height": self.canvas_area.background_image.shape[0]
                        if self.canvas_area.background_image is not None
                        else 0,
                        "background_path": self.current_background_path,
                    },
                    "instances": [],
                }

                # 添加所有素材实例的标注信息
                instances = self.layer_manager.get_all_instances()
                for i, instance in enumerate(instances):
                    if instance.visible:
                        x1, y1, x2, y2 = instance.get_bounding_rect()
                        instance_data = {
                            "id": i,
                            "material_name": instance.material_name,
                            "position": {"x": instance.x, "y": instance.y},
                            "scale": instance.scale,
                            "rotation": instance.rotation,
                            "layer_id": instance.layer_id,
                            "visible": instance.visible,
                            "blend_mode": getattr(instance, "blend_mode", "normal"),
                            "bounding_box": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1),
                            },
                        }
                        annotations["instances"].append(instance_data)

                # 保存JSON文件
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(annotations, f, ensure_ascii=False, indent=2)

                self.status_label.setText(f"标注已导出到: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出标注失败: {str(e)}")
                import traceback
                traceback.print_exc()

    def load_background_directory(self):
        """加载背景图像目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择背景图像目录")

        if directory:
            try:
                # 支持的图像格式
                image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

                # 获取目录中的所有图像文件
                image_files = []
                for file_path in Path(directory).iterdir():
                    if (
                        file_path.is_file()
                        and file_path.suffix.lower() in image_extensions
                    ):
                        image_files.append(str(file_path))

                if not image_files:
                    QMessageBox.information(self, "提示", "所选目录中没有找到图像文件")
                    return

                # 按文件名排序
                image_files.sort()
                self.background_images = image_files
                self.current_background_index = 0

                # 加载第一张图像
                self._load_background_by_index(0)
                self._update_background_navigation_buttons()

                self.status_label.setText(
                    f"加载背景目录成功，共 {len(image_files)} 张图像"
                )

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载背景目录失败: {str(e)}")

    def prev_background(self):
        """切换到上一张背景图像"""
        if self.background_images and self.current_background_index > 0:
            self.current_background_index -= 1
            self._load_background_by_index(self.current_background_index)
            self._update_background_navigation_buttons()

    def next_background(self):
        """切换到下一张背景图像"""
        if (
            self.background_images
            and self.current_background_index < len(self.background_images) - 1
        ):
            self.current_background_index += 1
            self._load_background_by_index(self.current_background_index)
            self._update_background_navigation_buttons()

    def _load_background_by_index(self, index: int):
        """根据索引加载背景图像"""
        if 0 <= index < len(self.background_images):
            file_path = self.background_images[index]
            self.current_background_index = index

            try:
                import cv2
                import numpy as np

                # 加载图像
                image = cv2.imdecode(
                    np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR
                )

                if image is not None:
                    # 自动清空所有素材
                    self.layer_manager.clear_all_layers()

                    # 清除画布选择
                    if hasattr(self.canvas_area, "graphics_view"):
                        self.canvas_area.graphics_view.scene.clearSelection()

                    # 设置新的背景图像
                    self.canvas_area.set_background_image(image)
                    self.current_background_path = file_path
                    self.is_modified = True
                    self.current_selected_material = None

                    # 更新取色器的背景图像
                    self.control_panel.property_widget.set_background_image(image)

                    # 适应窗口大小
                    self.canvas_area.zoom_to_fit()

                    # 更新界面
                    self._update_window_title()
                    self.control_panel.refresh_layers()
                    self.control_panel.set_current_instance(None)

                    # 更新状态栏显示当前图像信息
                    filename = Path(file_path).name
                    self.status_label.setText(
                        f"背景图像: {filename} ({index + 1}/{len(self.background_images)}) - 已清空素材并适应窗口"
                    )
                else:
                    QMessageBox.warning(self, "警告", f"无法加载图像文件: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载背景图像失败: {str(e)}")
                print(f"加载背景图像失败: {e}")
                import traceback

                traceback.print_exc()

    def _update_background_navigation_buttons(self):
        """更新背景图像导航按钮状态"""
        has_images = len(self.background_images) > 0
        can_go_prev = has_images and self.current_background_index > 0
        can_go_next = (
            has_images
            and self.current_background_index < len(self.background_images) - 1
        )

        self.prev_bg_action.setEnabled(can_go_prev)
        self.next_bg_action.setEnabled(can_go_next)

    def clear_all_materials(self):
        """清空所有素材"""
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空所有素材吗？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # 清空图层管理器中的所有实例
            self.layer_manager.clear_all_layers()

            # 清除画布选择
            self.canvas_area.selected_instance = None

            # 更新画布显示
            self.canvas_area.update_canvas()

            # 刷新控制面板
            self.control_panel.refresh_layers()
            self.control_panel.set_current_instance(None)

            # 标记为已修改
            self.is_modified = True
            self._update_window_title()

            self.status_label.setText("已清空所有素材")

    def show_random_generate_dialog(self):
        """显示随机生成物体对话框"""
        # 检查是否有背景图像
        if self.canvas_area.background_image is None:
            QMessageBox.warning(self, "警告", "请先加载背景图像")
            return

        # 检查是否有素材
        material_names = self.material_manager.get_material_names()
        if not material_names:
            QMessageBox.warning(self, "警告", "请先加载素材")
            return

        # 获取画布尺寸
        canvas_height, canvas_width = self.canvas_area.background_image.shape[:2]
        canvas_size = (canvas_width, canvas_height)

        # 显示对话框
        dialog = RandomGenerateDialog(
            material_names, canvas_size, self.canvas_area.background_image, self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_generation_params()
            self.generate_random_objects(params)

    def generate_random_objects(self, params: dict):
        """根据参数随机生成物体"""
        try:
            count = params["count"]
            mode = params["mode"]
            position_params = params["position"]
            rotation_params = params["rotation"]
            scale_params = params["scale"]
            blend_params = params["blend"]
            color_params = params["color_overlay"]

            # 获取素材列表
            available_materials = list(self.material_manager.materials.keys())
            if not available_materials:
                self.status_label.setText("没有可用的素材")
                return

            # 根据模式选择要使用的素材
            if mode == 0:  # 随机选择素材
                materials_to_use = [
                    random.choice(available_materials) for _ in range(count)
                ]
            elif mode == 1:  # 使用所有素材
                materials_to_use = (
                    available_materials * ((count // len(available_materials)) + 1)
                )[:count]
            elif mode == 2:  # 仅使用第一个素材
                materials_to_use = [available_materials[0]] * count
            elif mode == 3:  # 均匀分布所有素材
                materials_to_use = []
                per_material = count // len(available_materials)
                remainder = count % len(available_materials)
                for i, material in enumerate(available_materials):
                    material_count = per_material + (1 if i < remainder else 0)
                    materials_to_use.extend([material] * material_count)
                random.shuffle(materials_to_use)
            else:
                materials_to_use = [
                    random.choice(available_materials) for _ in range(count)
                ]

            # 生成位置
            positions = self._generate_positions(count, position_params)
            if not positions:
                self.status_label.setText("无法生成有效位置")
                return

            print(f"开始生成 {count} 个随机物体...")
            print(f"位置参数: {position_params}")
            print(f"混合模式参数: {blend_params}")
            print(f"色彩叠加参数: {color_params}")

            generated_count = 0
            failed_count = 0

            for i in range(min(count, len(positions), len(materials_to_use))):
                try:
                    material_name = materials_to_use[i]
                    x, y = positions[i]

                    # 生成旋转角度
                    if rotation_params["enabled"]:
                        rotation_min, rotation_max = rotation_params["range"]
                        rotation = random.uniform(rotation_min, rotation_max)
                    else:
                        rotation = 0.0

                    # 生成缩放比例
                    if scale_params["enabled"]:
                        scale_min, scale_max = scale_params["range"]
                        scale = random.uniform(scale_min, scale_max)
                    else:
                        scale = 1.0

                    # 选择混合模式
                    if blend_params["enabled"] and blend_params["modes"]:
                        blend_mode = random.choice(blend_params["modes"])
                    else:
                        blend_mode = "normal"

                    # 生成色彩叠加
                    color_overlay = None
                    overlay_opacity = 0.0

                    if color_params.get("enabled", False):
                        if color_params.get("use_background_colors", False):
                            # 使用背景色
                            color_overlay = self._generate_background_color_at_position(
                                x, y
                            )
                        elif color_params.get("use_preset", False):
                            # 使用预设颜色组
                            preset_group = color_params.get("preset_group", "暖色调")
                            color_overlay = self._generate_preset_color(preset_group)
                        else:
                            # 完全随机颜色
                            color_overlay = self._generate_random_color()

                        # 生成透明度
                        min_opacity, max_opacity = color_params.get(
                            "opacity_range", (0.2, 0.6)
                        )
                        overlay_opacity = random.uniform(min_opacity, max_opacity)

                    # 创建素材实例
                    instance = self.material_manager.create_instance(
                        material_name,
                        x,
                        y,
                        scale=scale,
                        rotation=rotation,
                        layer_id=self.layer_manager.current_layer_id,
                        blend_mode=blend_mode,
                        color_overlay=color_overlay,
                        overlay_opacity=overlay_opacity,
                    )

                    if instance:
                        self.layer_manager.add_instance_to_current_layer(instance)
                        generated_count += 1

                        # 输出详细信息（仅前5个）
                        if i < 5:
                            color_info = (
                                f", 颜色叠加: {color_overlay}, 透明度: {overlay_opacity:.2f}"
                                if color_overlay
                                else ""
                            )
                            print(
                                f"生成第 {i + 1} 个: {material_name}, 位置({x},{y}), 缩放{scale:.2f}, 旋转{rotation:.1f}°, 混合模式: {blend_mode}{color_info}"
                            )
                    else:
                        failed_count += 1
                        print(f"创建素材实例失败: {material_name}")

                except Exception as e:
                    failed_count += 1
                    print(f"生成第 {i + 1} 个物体时出错: {e}")

                # 更新进度（每10个更新一次界面）
                if (i + 1) % 10 == 0 or i == count - 1:
                    progress = min(100, int((i + 1) / count * 100))
                    self.status_label.setText(
                        f"生成进度: {progress}% ({i + 1}/{count})"
                    )
                    QApplication.processEvents()

            # 更新界面
            self.canvas_area.update_canvas()
            self.control_panel.refresh_layers()

            # 标记项目已修改
            self.is_modified = True
            self._update_window_title()

            # 显示结果
            result_msg = f"随机生成完成: 成功 {generated_count} 个"
            if failed_count > 0:
                result_msg += f", 失败 {failed_count} 个"

            if blend_params["enabled"]:
                result_msg += f", 使用混合模式: {', '.join(blend_params['modes'])}"

            if color_params["enabled"]:
                result_msg += f", 启用色彩叠加"

            self.status_label.setText(result_msg)
            print(result_msg)

        except Exception as e:
            error_msg = f"随机生成失败: {str(e)}"
            self.status_label.setText(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()

    def _generate_positions(
        self,
        count: int,
        position_params: dict,
    ) -> List[Tuple[int, int]]:
        """生成随机位置列表"""
        positions = []
        x_min, x_max = position_params["x_range"]
        y_min, y_max = position_params["y_range"]
        avoid_overlap = position_params["avoid_overlap"]

        # 确保范围有效
        if x_min >= x_max or y_min >= y_max:
            print(f"无效的位置范围: x({x_min}-{x_max}), y({y_min}-{y_max})")
            return []

        print(
            f"生成位置范围: x({x_min}-{x_max}), y({y_min}-{y_max}), 避免重叠: {avoid_overlap}"
        )

        if avoid_overlap:
            # 尝试避免重叠的位置生成
            min_distance = 50  # 最小距离
            max_attempts = count * 20  # 最大尝试次数
            attempts = 0

            while len(positions) < count and attempts < max_attempts:
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)

                # 检查是否与其他位置重叠
                if any(
                    (x - px) ** 2 + (y - py) ** 2 < min_distance**2
                    for px, py in positions
                ):
                    continue

                positions.append((x, y))
                attempts += 1

        print(f"实际生成位置数量: {len(positions)}")
        return positions

    def _generate_background_color_at_position(
        self, x: int, y: int
    ) -> Tuple[int, int, int]:
        """在指定位置提取背景颜色"""
        if self.canvas_area.background_image is None:
            return self._generate_random_color()

        try:
            # 获取背景图像
            bg_image = self.canvas_area.background_image
            h, w = bg_image.shape[:2]

            # 确保坐标在图像范围内
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            # 在位置周围取样（3x3区域）
            sample_size = 3
            colors = []

            for dy in range(-sample_size // 2, sample_size // 2 + 1):
                for dx in range(-sample_size // 2, sample_size // 2 + 1):
                    sample_x = max(0, min(x + dx, w - 1))
                    sample_y = max(0, min(y + dy, h - 1))

                    # 获取BGR颜色并转换为RGB
                    bgr_color = bg_image[sample_y, sample_x]
                    rgb_color = (
                        int(bgr_color[2]),
                        int(bgr_color[1]),
                        int(bgr_color[0]),
                    )
                    colors.append(rgb_color)

            # 随机选择一个采样颜色
            import random

            return random.choice(colors)

        except Exception as e:
            print(f"提取背景色失败: {e}")
            return self._generate_random_color()

    def _generate_random_color(self) -> Tuple[int, int, int]:
        """生成完全随机的颜色"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _generate_preset_color(self, preset_name: str) -> Tuple[int, int, int]:
        """根据预设名称生成颜色"""
        presets = {
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

        if preset_name in presets:
            return random.choice(presets[preset_name])
        else:
            return self._generate_random_color()

    def _toggle_incremental_update(self, checked: bool):
        """切换增量更新"""
        self.canvas_area.incremental_update_enabled = checked
        status_text = "增量更新已启用" if checked else "增量更新已禁用"
        self.status_label.setText(status_text)
        print(f"性能设置: {status_text}")


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("素材编辑器")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Material Editor")

    # 创建主窗口
    window = MaterialEditor()
    window.show()

    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
