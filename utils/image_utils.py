"""
图像处理工具
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def resize_image_keep_ratio(
    image: np.ndarray, max_width: int, max_height: int
) -> np.ndarray:
    """保持宽高比缩放图像"""
    h, w = image.shape[:2]

    # 计算缩放比例
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # 不放大

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def create_thumbnail(
    image: np.ndarray, size: Tuple[int, int] = (100, 100)
) -> np.ndarray:
    """创建缩略图"""
    thumb_w, thumb_h = size
    resized = resize_image_keep_ratio(image, thumb_w, thumb_h)

    # 创建白色背景
    thumbnail = np.ones((thumb_h, thumb_w, 3), dtype=np.uint8) * 255

    # 居中放置图像
    h, w = resized.shape[:2]
    y_offset = (thumb_h - h) // 2
    x_offset = (thumb_w - w) // 2

    thumbnail[y_offset : y_offset + h, x_offset : x_offset + w] = resized

    return thumbnail


def load_image_safe(image_path: str) -> Optional[np.ndarray]:
    """安全加载图像"""
    try:
        # 支持中文路径
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None


def save_image_safe(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """安全保存图像"""
    try:
        if output_path.lower().endswith((".jpg", ".jpeg")):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encoded_img = cv2.imencode(".jpg", image, encode_param)
        elif output_path.lower().endswith(".png"):
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            success, encoded_img = cv2.imencode(".png", image, encode_param)
        else:
            success, encoded_img = cv2.imencode(".jpg", image)

        if success:
            encoded_img.tofile(output_path)
            return True
        return False
    except Exception as e:
        print(f"保存图像失败 {output_path}: {e}")
        return False


def numpy_to_qimage(image: np.ndarray):
    """将numpy数组转换为QImage（需要在UI模块中使用）"""
    try:
        from PySide6.QtGui import QImage

        if len(image.shape) == 3:
            h, w, ch = image.shape
            bytes_per_line = ch * w

            # OpenCV使用BGR，Qt使用RGB
            if ch == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                )
            elif ch == 4:
                rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                return QImage(
                    rgba_image.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
                )
        elif len(image.shape) == 2:
            h, w = image.shape
            return QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
    except ImportError:
        print("PyQt6未安装，无法转换为QImage")
    except Exception as e:
        print(f"转换为QImage失败: {e}")

    return None


def qimage_to_numpy(qimage):
    """将QImage转换为numpy数组"""
    try:
        from PySide6.QtGui import QImage

        # 转换为RGB888格式
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)

        width = qimage.width()
        height = qimage.height()

        # 获取图像数据
        ptr = qimage.constBits()
        arr = np.array(ptr).reshape(height, width, 3)

        # Qt使用RGB，OpenCV使用BGR
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"转换为numpy数组失败: {e}")
        return None


def blend_images(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """混合两张图像"""
    if mask is not None:
        # 使用掩码混合
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 归一化掩码
        mask_norm = mask.astype(np.float32) / 255.0 * alpha
        mask_norm = np.expand_dims(mask_norm, axis=2)

        # 混合
        result = (
            background.astype(np.float32) * (1 - mask_norm)
            + foreground.astype(np.float32) * mask_norm
        )

        return result.astype(np.uint8)
    else:
        # 简单的alpha混合
        return cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)


def get_image_info(image: np.ndarray) -> dict:
    """获取图像信息"""
    if image is None:
        return {}

    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(image.dtype),
        "size_mb": image.nbytes / (1024 * 1024),
    }
