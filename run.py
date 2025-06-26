#!/usr/bin/env python3
"""
素材编辑器启动脚本
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def main():
    """主函数"""
    try:
        # 尝试导入PySide6
        from PySide6.QtWidgets import QApplication

        print("PySide6 导入成功")

        # 导入主程序
        from main import MaterialEditor

        # 创建应用程序
        app = QApplication(sys.argv)
        app.setApplicationName("素材编辑器")
        app.setApplicationVersion("1.0")
        app.setOrganizationName("MaterialEditor")

        # 创建主窗口
        window = MaterialEditor()
        window.show()

        print("素材编辑器启动成功！")
        print("使用说明：")
        print("1. 通过菜单 '文件 -> 加载背景图像' 加载背景图")
        print("2. 通过菜单 '文件 -> 加载素材目录' 加载素材")
        print("3. 在素材面板中选择素材，然后在画布上点击放置")
        print("4. 拖拽素材进行移动，使用鼠标滚轮缩放画布")
        print("5. 在属性面板中调整选中素材的属性")

        # 运行应用程序
        sys.exit(app.exec())

    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所需依赖:")
        print("pip install PyQt6 opencv-python numpy")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
