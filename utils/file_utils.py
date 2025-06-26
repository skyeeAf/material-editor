"""
文件处理工具
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_image_files(
    directory: Path, extensions: Optional[List[str]] = None
) -> List[Path]:
    """获取目录中的图像文件"""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    image_files = []
    if directory.exists() and directory.is_dir():
        for ext in extensions:
            image_files.extend(directory.rglob(f"*{ext}"))
            image_files.extend(directory.rglob(f"*{ext.upper()}"))

    return sorted(image_files)


def load_json_safe(json_path: Path) -> Optional[Dict[str, Any]]:
    """安全加载JSON文件"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败 {json_path}: {e}")
        return None


def save_json_safe(data: Dict[str, Any], json_path: Path) -> bool:
    """安全保存JSON文件"""
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"保存JSON文件失败 {json_path}: {e}")
        return False


def ensure_directory(path: Path) -> bool:
    """确保目录存在"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建目录失败 {path}: {e}")
        return False


def get_unique_filename(base_path: Path, extension: str = "") -> Path:
    """获取唯一的文件名（避免重复）"""
    if not extension.startswith(".") and extension:
        extension = "." + extension

    if not base_path.suffix and extension:
        base_path = base_path.with_suffix(extension)

    if not base_path.exists():
        return base_path

    # 文件已存在，添加数字后缀
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def validate_image_path(path: Path) -> bool:
    """验证图像路径是否有效"""
    if not path.exists():
        return False

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return path.suffix.lower() in valid_extensions


def get_file_size_mb(path: Path) -> float:
    """获取文件大小（MB）"""
    try:
        return path.stat().st_size / (1024 * 1024)
    except:
        return 0.0


def cleanup_temp_files(temp_dir: Path, pattern: str = "*"):
    """清理临时文件"""
    try:
        if temp_dir.exists() and temp_dir.is_dir():
            for file_path in temp_dir.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
    except Exception as e:
        print(f"清理临时文件失败: {e}")


def backup_file(file_path: Path, backup_dir: Optional[Path] = None) -> Optional[Path]:
    """备份文件"""
    try:
        if not file_path.exists():
            return None

        if backup_dir is None:
            backup_dir = file_path.parent / "backup"

        ensure_directory(backup_dir)

        # 生成备份文件名（包含时间戳）
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name

        # 复制文件
        import shutil

        shutil.copy2(file_path, backup_path)

        return backup_path
    except Exception as e:
        print(f"备份文件失败 {file_path}: {e}")
        return None


class FileManager:
    """文件管理器 - 处理项目文件的保存和加载"""

    def save_project(self, file_path: str, layer_manager) -> bool:
        """保存项目文件"""
        try:
            project_data = {"version": "1.0", "layers": layer_manager.to_dict()}
            return save_json_safe(project_data, Path(file_path))
        except Exception as e:
            print(f"保存项目失败: {e}")
            return False

    def load_project(self, file_path: str, layer_manager) -> bool:
        """加载项目文件"""
        try:
            project_data = load_json_safe(Path(file_path))
            if not project_data:
                return False

            # 这里可以根据需要实现项目数据的加载逻辑
            # 目前只是一个基本框架
            return True
        except Exception as e:
            print(f"加载项目失败: {e}")
            return False
