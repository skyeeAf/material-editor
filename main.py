#!/usr/bin/env python3
"""
素材编辑器主程序
基于PyQt6和OpenCV的图像合成工具
"""

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
from ui.canvas import CanvasScrollArea
from ui.controls import ControlPanel
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

        # 右侧画布区域
        self.canvas_area = CanvasScrollArea()
        self.canvas_area.canvas.set_layer_manager(self.layer_manager)
        splitter.addWidget(self.canvas_area)

        # 设置分割器比例
        splitter.setSizes([300, 1000])

    def _init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        # 加载背景图像
        load_bg_action = QAction("加载背景图像(&B)", self)
        load_bg_action.triggered.connect(self.load_background)
        file_menu.addAction(load_bg_action)

        # 加载素材目录
        load_materials_action = QAction("加载素材目录(&M)", self)
        load_materials_action.triggered.connect(self.load_materials)
        file_menu.addAction(load_materials_action)

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
        fit_action.triggered.connect(self.canvas_area.canvas.zoom_to_fit)
        view_menu.addAction(fit_action)

        # 实际大小
        actual_size_action = QAction("实际大小(&A)", self)
        actual_size_action.triggered.connect(
            self.canvas_area.canvas.zoom_to_actual_size
        )
        view_menu.addAction(actual_size_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")

        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _init_toolbar(self):
        """初始化工具栏"""
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)

        # 加载背景
        load_bg_action = QAction("加载背景", self)
        load_bg_action.triggered.connect(self.load_background)
        toolbar.addAction(load_bg_action)

        # 加载素材
        load_materials_action = QAction("加载素材", self)
        load_materials_action.triggered.connect(self.load_materials)
        toolbar.addAction(load_materials_action)

        toolbar.addSeparator()

        # 一键导出
        export_all_action = QAction("一键导出", self)
        export_all_action.triggered.connect(self.export_all)
        toolbar.addAction(export_all_action)

        toolbar.addSeparator()

        # 适应窗口
        fit_action = QAction("适应窗口", self)
        fit_action.triggered.connect(self.canvas_area.canvas.zoom_to_fit)
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
        self.canvas_area.canvas.instance_selected.connect(self._on_selection_changed)
        self.canvas_area.canvas.instance_moved.connect(self._on_canvas_modified)
        self.canvas_area.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas_area.canvas.canvas_double_clicked.connect(
            self._on_canvas_double_clicked
        )

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
        if self.add_mode == "click":
            self._add_material_at_position(x, y)
        else:
            # 双击模式下，单击只显示坐标
            self.status_label.setText(f"点击位置: ({x}, {y}) - 双击此位置可添加素材")

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
                    self.canvas_area.canvas.update_canvas()

                    # 刷新图层面板显示
                    self.control_panel.refresh_layers()

                    # 设置为选中状态
                    self.canvas_area.canvas.selected_instance = instance

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
        selected_instance = self.canvas_area.canvas.selected_instance
        if selected_instance is None:
            self.status_label.setText(
                "就绪 - 左键点击/拖动素材，右键拖动画布，滚轮以鼠标为中心缩放"
            )
        else:
            self.status_label.setText(
                f"已选择素材: {selected_instance.material_name} - 可拖动或使用快捷键调整"
            )
            self.control_panel.set_current_instance(selected_instance)

            # 自动选中对应的图层
            if selected_instance.layer_id:
                # 切换到图层标签页
                self.control_panel.set_active_tab("图层")
                # 在图层树中选中对应的图层
                self.control_panel.select_layer(selected_instance.layer_id)

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
                    self.canvas_area.canvas.update_canvas()
        except (ValueError, TypeError) as e:
            print(f"图层可见性改变失败: {e}, layer_id='{layer_id}'")

    def _on_property_changed(self, group: str, property_name: str, value):
        """素材属性改变事件"""
        try:
            selected_instance = self.canvas_area.canvas.selected_instance
            if selected_instance:
                # 更新素材实例属性
                if group == "position":
                    if property_name == "x":
                        selected_instance.x = value
                    elif property_name == "y":
                        selected_instance.y = value
                elif group == "transform":
                    if property_name == "scale":
                        selected_instance.scale = value
                        # 清除变换缓存以重新计算
                        selected_instance._transform_cache_key = None
                    elif property_name == "rotation":
                        selected_instance.rotation = value
                        # 清除变换缓存以重新计算
                        selected_instance._transform_cache_key = None
                elif group == "blend":
                    if property_name == "mode":
                        selected_instance.blend_mode = value

                # 更新画布显示
                self.canvas_area.canvas.update_canvas()

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
        scale = self.canvas_area.canvas.zoom_factor
        self.zoom_label.setText(f"缩放: {scale * 100:.0f}%")

    def _update_window_title(self):
        """更新窗口标题"""
        title = "素材编辑器"
        if self.current_background_path:
            title += f" - {Path(self.current_background_path).name}"
        if self.is_modified:
            title += " *"
        self.setWindowTitle(title)

    def load_background(self):
        """加载背景图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            try:
                import cv2
                import numpy as np

                image = cv2.imdecode(
                    np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is not None:
                    self.canvas_area.canvas.set_background_image(image)
                    self.current_background_path = file_path
                    self.is_modified = True
                    self.current_selected_material = None
                    self._update_window_title()
                    self.status_label.setText("背景图像加载成功")
                else:
                    QMessageBox.warning(self, "警告", "无法加载图像文件")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载背景图像失败: {str(e)}")

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
        if self.canvas_area.canvas.background_image is None:
            QMessageBox.warning(self, "警告", "请先加载背景图像")
            return

        if not self.layer_manager or not self.layer_manager.get_all_instances():
            QMessageBox.warning(self, "警告", "没有素材实例可导出")
            return

        # 让用户选择基础文件名
        base_file_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择导出文件基础名称",
            "",
            "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*.*)",
        )

        if not base_file_path:
            return

        try:
            # 获取基础文件名（不含扩展名）
            from pathlib import Path

            base_path = Path(base_file_path)
            base_name = base_path.stem
            base_dir = base_path.parent

            # 生成图像和标注文件路径
            image_path = base_dir / f"{base_name}.png"
            annotation_path = base_dir / f"{base_name}.json"

            # 导出图像
            import cv2

            canvas_image = self.canvas_area.canvas.background_image.copy()

            # 绘制所有可见的素材实例
            instances = self.layer_manager.get_all_instances()
            for instance in instances:
                if instance.visible:
                    canvas_image = self.canvas_area.canvas._draw_instance_on_canvas(
                        canvas_image, instance
                    )

            # 保存图像
            success = cv2.imwrite(str(image_path), canvas_image)

            if not success:
                QMessageBox.critical(self, "错误", "导出图像失败")
                return

            # 导出标注
            import json
            import os

            # 构建标注数据 - 使用AnyMark格式
            annotations = {
                "info": {
                    "description": "AnyMark",
                    "folder": str(base_dir),
                    "name": image_path.name,
                    "width": self.canvas_area.canvas.background_image.shape[1],
                    "height": self.canvas_area.canvas.background_image.shape[0],
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

            self.status_label.setText(
                f"导出成功：图像({image_path.name})、标注({annotation_path.name})"
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
        selected_instance = self.canvas_area.canvas.selected_instance
        if selected_instance:
            # 从图层管理器中删除实例
            self.layer_manager.remove_instance(selected_instance)
            # 清除画布选择
            self.canvas_area.canvas.selected_instance = None
            # 更新画布显示
            self.canvas_area.canvas.update_canvas()
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
            self.canvas_area.canvas.update_canvas()
            self.is_modified = True
            self._update_window_title()
        except Exception as e:
            print(f"素材实例可见性改变失败: {e}")

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # Delete键删除选中的素材实例
        if event.key() == Qt.Key.Key_Delete:
            selected_instance = self.canvas_area.canvas.selected_instance
            if selected_instance:
                # 从图层管理器中删除实例
                self.layer_manager.remove_instance(selected_instance)
                # 清除画布选择
                self.canvas_area.canvas.selected_instance = None
                # 更新画布显示
                self.canvas_area.canvas.update_canvas()
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
        selected_instance = self.canvas_area.canvas.selected_instance
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
                self.canvas_area.canvas.update_canvas()
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
            selected_instance = self.canvas_area.canvas.selected_instance
            if selected_instance:
                selected_instance._transform_cache_key = None

    def export_image(self):
        """导出图像（兼容性方法）"""
        if self.canvas_area.canvas.background_image is None:
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
                canvas_image = self.canvas_area.canvas.background_image.copy()

                # 绘制所有可见的素材实例
                if self.layer_manager:
                    instances = self.layer_manager.get_all_instances()
                    for instance in instances:
                        if instance.visible:
                            canvas_image = (
                                self.canvas_area.canvas._draw_instance_on_canvas(
                                    canvas_image, instance
                                )
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
                        "width": self.canvas_area.canvas.background_image.shape[1]
                        if self.canvas_area.canvas.background_image is not None
                        else 0,
                        "height": self.canvas_area.canvas.background_image.shape[0]
                        if self.canvas_area.canvas.background_image is not None
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
