"""libcom 图像协调化子进程工作脚本（不导入 PySide6，避免 typing.Self 冲突）.

由 harmonize 模块通过 subprocess 调用，也可手动运行预下载权重。

用法:
    # 预下载模型权重（不启动推理）
    python _harmonize_worker.py --download
    python _harmonize_worker.py --download --model-dir D:/models/libcom

    # 指定权重目录后启动推理（由父进程自动调用）
    set LIBCOM_MODEL_DIR=D:/models/libcom
    python _harmonize_worker.py

协议 (stdin/stdout 二进制管道):
    输入 (stdin):
        4 bytes  H   (uint32 little-endian)
        4 bytes  W   (uint32 little-endian)
        H*W*3 bytes  BGR 合成图 (uint8)
        H*W bytes    前景掩码 (uint8)
    输出 (stdout):
        4 bytes  H'  (uint32 little-endian)
        4 bytes  W'  (uint32 little-endian)
        H'*W'*3 bytes BGR 协调结果 (uint8)
"""

import argparse
import importlib.util
import os
import struct
import sys
import types

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 用空壳 libcom 包替换 __init__.py，避免导入所有子模块的重量级依赖。
# 只加载我们需要的 image_harmonization 子模块。
# ---------------------------------------------------------------------------
_spec = importlib.util.find_spec("libcom")
if _spec is None:
    print("libcom not installed", file=sys.stderr)
    sys.exit(1)
_mod = types.ModuleType("libcom")
_mod.__path__ = list(_spec.submodule_search_locations)  # type: ignore
_mod.__package__ = "libcom"
sys.modules["libcom"] = _mod

from libcom.image_harmonization import ImageHarmonizationModel  # type: ignore

# 第三方依赖:  pip install libcom einops
# 需要 GPU (CUDA)

_model: ImageHarmonizationModel | None = None


def _load_model() -> ImageHarmonizationModel:
    """加载 PCTNet 模型（首次调用时触发下载）.

    Returns:
        ImageHarmonizationModel: 加载好的模型实例。
    """
    global _model
    if _model is not None:
        return _model

    model_dir = os.environ.get("LIBCOM_MODEL_DIR",
                               os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    # 确保 LIBCOM_MODEL_DIR 已设置，供 libcom 内部 model_dir 读取
    os.environ["LIBCOM_MODEL_DIR"] = model_dir
    print(f"[harmonize_worker] 权重目录: {model_dir}", file=sys.stderr)

    print("[harmonize_worker] 正在加载 PCTNet 模型…", file=sys.stderr)
    _model = ImageHarmonizationModel(device=0, model_type="PCTNet")
    print("[harmonize_worker] 模型加载完成 ✓", file=sys.stderr)
    return _model


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
    """从 stdin 读取合成图和掩码 → libcom 协调 → 结果写入 stdout。"""
    model = _load_model()

    sin = sys.stdin.buffer
    sout = sys.stdout.buffer

    # ---- 读取输入 ----
    header = _read_exact(sin, 8)
    h, w = struct.unpack("<II", header)

    img_bytes = _read_exact(sin, h * w * 3)
    composite_bgr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, 3)

    mask_bytes = _read_exact(sin, h * w)
    fg_mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(h, w)

    # ---- libcom 协调 ----
    # libcom 接受 BGR ndarray + mask ndarray
    result_bgr = model(composite_bgr, fg_mask)

    # 确保结果是 uint8 BGR
    if result_bgr.dtype != np.uint8:
        result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
    if result_bgr.ndim == 2:
        result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_GRAY2BGR)

    # ---- 写出结果 ----
    rh, rw = result_bgr.shape[:2]
    sout.write(struct.pack("<II", rh, rw))
    sout.write(result_bgr.tobytes())
    sout.flush()


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="libcom 协调化工作进程 / 权重预下载工具"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="仅下载模型权重并退出，不启动推理服务。"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="指定模型权重存放目录（覆盖 LIBCOM_MODEL_DIR 环境变量）。"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # --model-dir 优先级最高，覆盖环境变量
    if args.model_dir:
        os.environ["LIBCOM_MODEL_DIR"] = args.model_dir
        print(f"[harmonize_worker] --model-dir 覆盖: {args.model_dir}",
              file=sys.stderr)

    if args.download:
        # 触发模型加载（含权重下载），然后退出
        _load_model()
        print("[harmonize_worker] 模型权重下载/验证完成，退出。")
        sys.exit(0)

    main()
