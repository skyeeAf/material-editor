"""图像协调化与接缝修复模块.

提供两种色调协调后端:
    1. libcom ImageHarmonizationModel (GPU, 效果好):  pip install libcom
       注: 因 PySide6 与 torch 的 typing.Self 冲突，通过子进程调用。
    2. Reinhard 色调匹配 (纯 numpy/cv2, 无额外依赖)

接缝修复:
    利用形态学提取素材边缘 mask，调用 patchmatch_inpaint 修复接缝。
"""

import os
import struct
import subprocess
import sys
import time
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 后端枚举与检测
# ---------------------------------------------------------------------------


class HarmonizeBackend(Enum):
    """色调协调后端标识。"""

    AUTO = "auto"
    LIBCOM = "libcom"
    REINHARD = "reinhard"


_HAS_LIBCOM: bool | None = None  # None = 尚未检测
_HARMONIZE_WORKER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_harmonize_worker.py",
)

# 可通过环境变量 LIBCOM_MODEL_DIR 指定权重目录，默认为项目目录下的 ./models/。
# 预下载命令（在终端执行）：
#   python _harmonize_worker.py --download [--model-dir <DIR>]
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models",
)
_libcom_model_dir: str = os.environ.get("LIBCOM_MODEL_DIR", _DEFAULT_MODEL_DIR)


def set_libcom_model_dir(path: str | None) -> None:
    """设置 libcom 权重目录（在首次推理之前调用）.

    Args:
        path (str | None): 权重目录路径。传 None 恢复默认（项目 ./models/ 目录）。

    Note:
        此设置仅影响后续的子进程调用，不会修改系统环境变量。
    """
    global _libcom_model_dir
    _libcom_model_dir = path if path else _DEFAULT_MODEL_DIR
    print(f"[Harmonize] 权重目录已设置: {_libcom_model_dir}")


def get_libcom_model_dir() -> str:
    """返回当前配置的 libcom 权重目录.

    Returns:
        str: 当前设置的路径。
    """
    return _libcom_model_dir


def _make_worker_env() -> dict:
    """构建传递给子进程的环境变量字典.

    Returns:
        dict: 包含 LIBCOM_MODEL_DIR 的 env dict。
    """
    env = os.environ.copy()
    env["LIBCOM_MODEL_DIR"] = _libcom_model_dir
    return env


def _check_libcom() -> bool:
    """懒检测 libcom 是否可用（子进程探测，避免 typing 冲突）。"""
    global _HAS_LIBCOM
    if _HAS_LIBCOM is not None:
        return _HAS_LIBCOM
    if not os.path.isfile(_HARMONIZE_WORKER):
        print("[Harmonize] libcom 不可用: _harmonize_worker.py 不存在")
        _HAS_LIBCOM = False
        return False
    try:
        # 用空壳 libcom 包绕过 __init__.py（它会导入所有子模块，
        # 每个都有各自的重量级依赖），只加载我们需要的 image_harmonization
        _detect_code = (
            "import importlib.util,types,sys,os;"
            "spec=importlib.util.find_spec('libcom');"
            "assert spec,'libcom not installed';"
            "mod=types.ModuleType('libcom');"
            "mod.__path__=list(spec.submodule_search_locations);"
            "mod.__package__='libcom';"
            "sys.modules['libcom']=mod;"
            "from libcom.image_harmonization import "
            "ImageHarmonizationModel;"
            "d=os.environ.get('LIBCOM_MODEL_DIR','(libcom 包安装目录)');"
            "print('OK|'+d)"
        )
        ret = subprocess.run(
            [sys.executable, "-c", _detect_code],
            capture_output=True, text=True, timeout=30,
            env=_make_worker_env(),
        )
        ok_line = ret.stdout.strip()
        _HAS_LIBCOM = ret.returncode == 0 and ok_line.startswith("OK")
        if _HAS_LIBCOM:
            model_path = ok_line.split("|", 1)[1] if "|" in ok_line else "未知"
            print(f"[Harmonize] libcom (GPU) 可用 ✓ | 权重目录: {model_path}")
        else:
            print(f"[Harmonize] libcom 不可用: {ret.stderr.strip()}")
    except Exception as _e:
        print(f"[Harmonize] libcom 检测失败: {_e}")
        _HAS_LIBCOM = False
    return _HAS_LIBCOM


_forced_backend: HarmonizeBackend = HarmonizeBackend.AUTO

_BACKEND_DISPLAY: Dict[HarmonizeBackend, str] = {
    HarmonizeBackend.AUTO: "自动",
    HarmonizeBackend.LIBCOM: "libcom (GPU)",
    HarmonizeBackend.REINHARD: "Reinhard (CPU)",
}


def get_available_harmonize_backends() -> List[Tuple[HarmonizeBackend, str, bool]]:
    """返回所有协调化后端及其可用状态。

    Returns:
        list[tuple[HarmonizeBackend, str, bool]]: (枚举值, 显示名, 是否可用).
    """
    _check_libcom()
    return [
        (HarmonizeBackend.AUTO, _BACKEND_DISPLAY[HarmonizeBackend.AUTO], True),
        (HarmonizeBackend.LIBCOM, _BACKEND_DISPLAY[HarmonizeBackend.LIBCOM],
         bool(_HAS_LIBCOM)),
        (HarmonizeBackend.REINHARD,
         _BACKEND_DISPLAY[HarmonizeBackend.REINHARD], True),
    ]


def set_harmonize_backend(backend: HarmonizeBackend) -> None:
    """强制指定协调化后端。

    Args:
        backend (HarmonizeBackend): 要使用的后端.

    Raises:
        ValueError: 指定的后端不可用.
    """
    global _forced_backend
    if backend == HarmonizeBackend.LIBCOM and not _check_libcom():
        raise ValueError("libcom 后端不可用。")
    _forced_backend = backend


def _resolve_harmonize_backend() -> HarmonizeBackend:
    """解析实际使用的协调化后端。"""
    if _forced_backend != HarmonizeBackend.AUTO:
        return _forced_backend
    if _check_libcom():
        return HarmonizeBackend.LIBCOM
    return HarmonizeBackend.REINHARD


# ===================================================================
# 公开入口：色调协调
# ===================================================================

def harmonize_region(
    composite: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """对合成图中的前景区域进行色调协调。

    Args:
        composite (np.ndarray): BGR uint8 合成图（素材已粘贴上去）.
        fg_mask (np.ndarray): 前景 mask, 255=前景, 0=背景, uint8.

    Returns:
        np.ndarray: 协调后的 BGR uint8 图像.
    """
    if not np.any(fg_mask > 0):
        return composite.copy()

    fg_count = int(np.count_nonzero(fg_mask > 0))
    backend = _resolve_harmonize_backend()
    print(f"[Harmonize] 开始 | 后端: {_BACKEND_DISPLAY[backend]}, "
          f"前景像素: {fg_count}")

    t0 = time.perf_counter()
    if backend == HarmonizeBackend.LIBCOM:
        result = _harmonize_libcom(composite, fg_mask)
    else:
        result = _harmonize_reinhard(composite, fg_mask)
    elapsed = time.perf_counter() - t0

    # 变化量：前景区域的平均像素差
    fg_bin = fg_mask > 127
    diff = np.abs(result[fg_bin].astype(np.float32)
                  - composite[fg_bin].astype(np.float32))
    mean_diff = float(diff.mean()) if diff.size > 0 else 0.0
    print(f"[Harmonize] 完成 | 耗时: {elapsed:.2f}s, "
          f"平均像素变化: {mean_diff:.1f}/255")
    return result


# ===================================================================
# Reinhard 色调匹配（零依赖）
# ===================================================================

def _harmonize_reinhard(
    composite: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Reinhard 色调匹配: 将前景 LAB 统计量对齐到周围背景。

    Args:
        composite (np.ndarray): BGR uint8.
        fg_mask (np.ndarray): 前景 mask uint8.

    Returns:
        np.ndarray: BGR uint8 结果.
    """
    fg_bin = fg_mask > 127

    # 参考区域：前景 mask 膨胀后的环形带（排除前景本身）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    dilated = cv2.dilate(fg_mask, kernel) > 127
    ref_bin = dilated & (~fg_bin)

    # 若参考区域不足，退回整个非前景区域
    if np.count_nonzero(ref_bin) < 100:
        ref_bin = ~fg_bin
    if np.count_nonzero(ref_bin) < 10 or np.count_nonzero(fg_bin) < 10:
        return composite.copy()

    lab = cv2.cvtColor(composite, cv2.COLOR_BGR2LAB).astype(np.float32)

    fg_pixels = lab[fg_bin]   # (N, 3)
    ref_pixels = lab[ref_bin]  # (M, 3)

    fg_mean = fg_pixels.mean(axis=0)
    fg_std = fg_pixels.std(axis=0) + 1e-6
    ref_mean = ref_pixels.mean(axis=0)
    ref_std = ref_pixels.std(axis=0) + 1e-6

    result_lab = lab.copy()
    for ch in range(3):
        vals = result_lab[fg_bin, ch]
        result_lab[fg_bin, ch] = (vals - fg_mean[ch]) * (ref_std[ch] / fg_std[ch]) + ref_mean[ch]

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    output = composite.copy()
    output[fg_bin] = result[fg_bin]
    return output


# ===================================================================
# libcom 子进程协调化
# ===================================================================

_libcom_first_run = True


def _harmonize_libcom(
    composite: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """通过子进程调用 libcom ImageHarmonizationModel（管道传输）。

    Args:
        composite (np.ndarray): BGR uint8.
        fg_mask (np.ndarray): 前景 mask uint8.

    Returns:
        np.ndarray: BGR uint8 结果.

    Raises:
        RuntimeError: 子进程失败.
    """
    global _libcom_first_run
    if _libcom_first_run:
        print("[Harmonize] libcom 首次推理，需初始化 CUDA + 加载模型，"
              "可能需要 30~60 秒，请耐心等待…")
        _libcom_first_run = False

    h, w = composite.shape[:2]

    input_data = (
        struct.pack("<II", h, w)
        + composite.tobytes()
        + fg_mask.tobytes()
    )

    # stderr=None → 继承父进程 stderr，worker 的日志实时显示在控制台
    proc = subprocess.Popen(
        [sys.executable, "-u", _HARMONIZE_WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        env=_make_worker_env(),
    )
    out, _ = proc.communicate(input=input_data, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"libcom 子进程失败 (exitcode={proc.returncode})")

    if len(out) < 8:
        raise RuntimeError("libcom 子进程未返回有效数据")

    rh, rw = struct.unpack("<II", out[:8])
    expected = 8 + rh * rw * 3
    if len(out) != expected:
        raise RuntimeError(
            f"libcom 输出数据长度不匹配: 预期 {expected}, 实际 {len(out)}"
        )
    result = np.frombuffer(out, dtype=np.uint8, offset=8).reshape(rh, rw, 3)
    return result.copy()


# ===================================================================
# 公开入口：接缝修复
# ===================================================================

def repair_seam(
    composite: np.ndarray,
    fg_mask: np.ndarray,
    edge_width: int = 5,
    patch_size: int = 7,
) -> np.ndarray:
    """修复素材粘贴后的边缘接缝。

    提取前景 mask 边缘的窄带 mask，用 inpaint 修复该区域。

    Args:
        composite (np.ndarray): BGR uint8 合成图.
        fg_mask (np.ndarray): 前景 mask, 255=前景, uint8.
        edge_width (int, optional): 边缘带宽度（像素）, 默认 5.
        patch_size (int, optional): inpaint patch 大小, 默认 7.

    Returns:
        np.ndarray: 修复后的 BGR uint8 图像.
    """
    from patchmatch_inpaint import patchmatch_inpaint, _get_backend_name

    if not np.any(fg_mask > 0):
        return composite.copy()

    k = max(1, edge_width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
    dilated = cv2.dilate(fg_mask, kernel)
    eroded = cv2.erode(fg_mask, kernel)
    edge_mask = cv2.subtract(dilated, eroded)

    if not np.any(edge_mask > 0):
        return composite.copy()

    edge_pixels = int(np.count_nonzero(edge_mask > 0))
    print(f"[Seam Repair] 开始 | 边缘宽度: {edge_width}px, "
          f"边缘像素: {edge_pixels}, "
          f"inpaint 后端: {_get_backend_name()}")

    t0 = time.perf_counter()
    result = patchmatch_inpaint(composite, edge_mask, patch_size=patch_size)
    elapsed = time.perf_counter() - t0

    edge_bin = edge_mask > 127
    diff = np.abs(result[edge_bin].astype(np.float32)
                  - composite[edge_bin].astype(np.float32))
    mean_diff = float(diff.mean()) if diff.size > 0 else 0.0
    print(f"[Seam Repair] 完成 | 耗时: {elapsed:.2f}s, "
          f"平均像素变化: {mean_diff:.1f}/255")
    return result
