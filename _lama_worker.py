"""LaMa 推理子进程工作脚本（不导入 PySide6，避免 typing.Self 冲突）.

由 patchmatch_inpaint 模块通过 subprocess 调用，不要直接运行。

协议 (stdin/stdout 二进制管道):
    输入 (stdin):
        4 bytes  H   (uint32 little-endian)
        4 bytes  W   (uint32 little-endian)
        H*W*3 bytes  BGR 图像 (uint8)
        H*W bytes    灰度掩码 (uint8)
    输出 (stdout):
        4 bytes  H'  (uint32 little-endian)
        4 bytes  W'  (uint32 little-endian)
        H'*W'*3 bytes BGR 结果 (uint8)
"""

import os
import struct
import sys

import cv2
import numpy as np
from PIL import Image

# 打印权重路径信息
_model_path = os.environ.get("LAMA_MODEL", "(torch hub 默认缓存)")
print(f"[lama_worker] 权重路径: {_model_path}", file=sys.stderr)
print("[lama_worker] 正在加载 LaMa 模型…", file=sys.stderr)

from simple_lama_inpainting import SimpleLama  # type: ignore

_lama = SimpleLama()
print("[lama_worker] 模型加载完成 ✓", file=sys.stderr)


def _read_exact(stream, n: int) -> bytes:
    """从流中精确读取 n 字节。"""
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"预期 {n} 字节，仅读到 {len(buf)} 字节")
        buf += chunk
    return buf


def main() -> None:
    """从 stdin 读取图像和掩码 → LaMa 推理 → 结果写入 stdout。"""
    sin = sys.stdin.buffer
    sout = sys.stdout.buffer

    # ---- 读取输入 ----
    header = _read_exact(sin, 8)
    h, w = struct.unpack("<II", header)

    img_bytes = _read_exact(sin, h * w * 3)
    img_bgr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, 3)

    mask_bytes = _read_exact(sin, h * w)
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(h, w)

    # ---- LaMa 推理 ----
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_mask = Image.fromarray(mask)

    result_pil = _lama(pil_img, pil_mask)

    result_rgb = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    # ---- 写出结果 ----
    rh, rw = result_bgr.shape[:2]
    sout.write(struct.pack("<II", rh, rw))
    sout.write(result_bgr.tobytes())
    sout.flush()


if __name__ == "__main__":
    main()
