"""内容识别填充（Content-Aware Fill）.

三级自动回退:
    1. simple-lama-inpainting (深度学习, 效果最好+快速):  pip install simple-lama-inpainting
       注: 因 PySide6 与 torch 的 typing.Self 冲突，LaMa 通过子进程调用。
    2. pypatchmatch (C++ PatchMatch, 快速):               pip install pypatchmatch
    3. 纯 Python 多尺度 PatchMatch (无需额外依赖, 较慢)
"""

import os
import struct
import subprocess
import sys
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 后端枚举与管理
# ---------------------------------------------------------------------------


class InpaintBackend(Enum):
    """可用的 inpaint 后端标识。"""

    AUTO = "auto"
    LAMA = "lama"
    PYPATCHMATCH = "pypatchmatch"
    PYTHON_FALLBACK = "python_fallback"


_HAS_LAMA: bool | None = None  # None = 尚未检测
_LAMA_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_lama_worker.py")

# LaMa 模型权重路径，默认为项目 ./models/big-lama.pt
_DEFAULT_LAMA_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "big-lama.pt",
)
_lama_model_path: str = os.environ.get("LAMA_MODEL", _DEFAULT_LAMA_MODEL)


def set_lama_model_path(path: str | None) -> None:
    """设置 LaMa 模型权重路径（在首次推理之前调用）.

    Args:
        path (str | None): 权重文件路径。传 None 恢复默认（项目 ./models/big-lama.pt）。
    """
    global _lama_model_path
    _lama_model_path = path if path else _DEFAULT_LAMA_MODEL
    print(f"[Content-Aware Fill] LaMa 权重路径已设置: {_lama_model_path}")


def get_lama_model_path() -> str:
    """返回当前配置的 LaMa 模型权重路径.

    Returns:
        str: 当前设置的路径。
    """
    return _lama_model_path


def _make_lama_env() -> dict:
    """构建传递给 LaMa 子进程的环境变量字典.

    Returns:
        dict: 包含 LAMA_MODEL 的 env dict。
    """
    env = os.environ.copy()
    env["LAMA_MODEL"] = _lama_model_path
    return env


def _check_lama() -> bool:
    """懒检测 simple-lama-inpainting 是否可用（子进程探测，避免 typing 冲突）。"""
    global _HAS_LAMA
    if _HAS_LAMA is not None:
        return _HAS_LAMA
    if not os.path.isfile(_LAMA_WORKER):
        print("[Content-Aware Fill] LaMa 不可用: _lama_worker.py 不存在")
        _HAS_LAMA = False
        return False
    try:
        ret = subprocess.run(
            [sys.executable, "-c",
             "from simple_lama_inpainting import SimpleLama; print('OK')"],
            capture_output=True, text=True, timeout=30,
        )
        _HAS_LAMA = ret.returncode == 0 and "OK" in ret.stdout
        if _HAS_LAMA:
            exists = os.path.isfile(_lama_model_path)
            status = "✓" if exists else "✗ (文件不存在，首次推理将自动下载)"
            print(f"[Content-Aware Fill] LaMa 可用 | 权重: {_lama_model_path} {status}")
        else:
            print(f"[Content-Aware Fill] LaMa 不可用: {ret.stderr.strip()}")
    except Exception as _e:
        print(f"[Content-Aware Fill] LaMa 检测失败: {_e}")
        _HAS_LAMA = False
    return _HAS_LAMA


_HAS_PYPATCHMATCH = False
_pm_mod = None

try:
    from patchmatch import patch_match as _pm_mod  # type: ignore
    _HAS_PYPATCHMATCH = True
except Exception as _e:
    print(f"[Content-Aware Fill] pypatchmatch 不可用: {_e}")
    _HAS_PYPATCHMATCH = False


# 当前强制后端；None 表示自动选择
_forced_backend: InpaintBackend = InpaintBackend.AUTO


# 后端显示名称映射
_BACKEND_DISPLAY: Dict[InpaintBackend, str] = {
    InpaintBackend.AUTO: "自动",
    InpaintBackend.LAMA: "LaMa (deep learning)",
    InpaintBackend.PYPATCHMATCH: "pypatchmatch (C++)",
    InpaintBackend.PYTHON_FALLBACK: "Python PatchMatch (fallback)",
}


def get_available_backends() -> List[Tuple[InpaintBackend, str, bool]]:
    """返回所有后端及其可用状态。

    Returns:
        list[tuple[InpaintBackend, str, bool]]: (枚举值, 显示名, 是否可用) 列表.
    """
    _check_lama()
    return [
        (InpaintBackend.AUTO, _BACKEND_DISPLAY[InpaintBackend.AUTO], True),
        (InpaintBackend.LAMA, _BACKEND_DISPLAY[InpaintBackend.LAMA],
         bool(_HAS_LAMA)),
        (InpaintBackend.PYPATCHMATCH,
         _BACKEND_DISPLAY[InpaintBackend.PYPATCHMATCH],
         _HAS_PYPATCHMATCH),
        (InpaintBackend.PYTHON_FALLBACK,
         _BACKEND_DISPLAY[InpaintBackend.PYTHON_FALLBACK],
         True),
    ]


def set_backend(backend: InpaintBackend) -> None:
    """强制指定 inpaint 后端。

    Args:
        backend (InpaintBackend): 要使用的后端.

    Raises:
        ValueError: 指定的后端不可用.
    """
    global _forced_backend
    if backend == InpaintBackend.LAMA and not _check_lama():
        raise ValueError("LaMa 后端不可用。")
    if backend == InpaintBackend.PYPATCHMATCH and not _HAS_PYPATCHMATCH:
        raise ValueError("pypatchmatch 后端不可用。")
    _forced_backend = backend


def get_backend() -> InpaintBackend:
    """返回当前设定的后端。

    Returns:
        InpaintBackend: 当前后端枚举值.
    """
    return _forced_backend


def _get_backend_name() -> str:
    """返回实际将要使用的后端名称（触发懒检测）。"""
    backend = _resolve_backend()
    return _BACKEND_DISPLAY[backend]


def _resolve_backend() -> InpaintBackend:
    """根据 _forced_backend 和可用性，解析出实际使用的后端。"""
    if _forced_backend != InpaintBackend.AUTO:
        return _forced_backend
    if _check_lama():
        return InpaintBackend.LAMA
    if _HAS_PYPATCHMATCH:
        return InpaintBackend.PYPATCHMATCH
    return InpaintBackend.PYTHON_FALLBACK


# ===================================================================
# 公开入口
# ===================================================================

def patchmatch_inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 7,
    max_size: int = 0,
) -> np.ndarray:
    """内容识别填充。

    对所有后端统一执行 ROI 裁剪 → 可选降采样 → 后端填充 → 上采样 → 边界融合 → 回贴。

    Args:
        img (np.ndarray): BGR 输入图像, uint8.
        mask (np.ndarray): 填充掩码, 255=待填充, 0=已知, uint8.
        patch_size (int, optional): patch 边长（奇数）, 默认 7.
        max_size (int, optional): ROI 长边上限（像素）, 0=不限制, 默认 0.

    Returns:
        np.ndarray: 填充结果, BGR uint8.
    """
    assert img.ndim == 3 and img.shape[2] == 3, "需要 BGR 3 通道图像。"
    assert mask.ndim == 2, "掩码需要单通道。"

    mask_bin = (mask > 127).astype(np.uint8) * 255
    if not np.any(mask_bin > 0):
        return img.copy()

    # ---- ROI 裁剪 ----
    padding = max(patch_size * 10, 80)
    crop_img, crop_mask, y0, x0, y1, x1 = _crop_work_area(
        img, mask_bin, padding,
    )
    ch, cw = crop_img.shape[:2]
    full_h, full_w = img.shape[:2]
    print(f"[Content-Aware Fill] ROI 裁剪: {full_w}×{full_h} → {cw}×{ch}")

    # ---- 可选降采样 ----
    scale = 1.0
    if max_size > 0 and max(ch, cw) > max_size:
        scale = max_size / max(ch, cw)
        new_h = max(1, int(ch * scale))
        new_w = max(1, int(cw * scale))
        work_img = cv2.resize(crop_img, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
        work_mask = cv2.resize(crop_mask, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        work_mask = (work_mask > 127).astype(np.uint8) * 255
        print(f"[Content-Aware Fill] 降采样: {cw}×{ch} → {new_w}×{new_h} "
              f"(scale={scale:.2f})")
    else:
        work_img = crop_img
        work_mask = crop_mask

    # ---- 调用后端 ----
    wh, ww = work_img.shape[:2]
    backend = _resolve_backend()
    if backend == InpaintBackend.LAMA:
        filled = _inpaint_lama(work_img, work_mask)
    elif backend == InpaintBackend.PYPATCHMATCH:
        filled = _inpaint_native(work_img, work_mask, patch_size)
    else:
        filled = _inpaint_fallback(work_img, work_mask, patch_size)

    # 某些后端（如 LaMa）会内部 pad 导致输出尺寸不一致，强制裁回
    if filled.shape[:2] != (wh, ww):
        filled = filled[:wh, :ww]

    # ---- 上采样回 crop 尺寸 ----
    if scale < 1.0:
        filled = cv2.resize(filled, (cw, ch),
                            interpolation=cv2.INTER_LANCZOS4)

    # ---- 边界融合 + 回贴 ----
    blended = _blend_seam(crop_img, filled, crop_mask, width=patch_size)
    result = img.copy()
    result[y0:y1, x0:x1] = blended
    return result


# ===================================================================
# 方案 A：LaMa 深度学习 - 子进程隔离 (最高优先级)
# ===================================================================

def _inpaint_lama(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """通过子进程管道调用 LaMa 模型执行填充（避免 PySide6/torch 冲突）。

    数据通过 stdin/stdout 二进制管道传输，无磁盘 I/O。

    Args:
        img (np.ndarray): BGR uint8.
        mask (np.ndarray): 二值掩码 uint8.

    Returns:
        np.ndarray: BGR uint8 结果.

    Raises:
        RuntimeError: 子进程执行失败.
    """
    h, w = img.shape[:2]

    # 构造输入二进制包：header(H, W) + BGR数据 + mask数据
    input_data = struct.pack("<II", h, w) + img.tobytes() + mask.tobytes()

    # stderr=None → 继承父进程 stderr，worker 日志实时显示在控制台
    proc = subprocess.Popen(
        [sys.executable, "-u", _LAMA_WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        env=_make_lama_env(),
    )
    out, _ = proc.communicate(input=input_data, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"LaMa 子进程失败 (exitcode={proc.returncode})")

    if len(out) < 8:
        raise RuntimeError("LaMa 子进程未返回有效数据")

    rh, rw = struct.unpack("<II", out[:8])
    expected = 8 + rh * rw * 3
    if len(out) != expected:
        raise RuntimeError(
            f"LaMa 输出数据长度不匹配: 预期 {expected}, 实际 {len(out)}"
        )
    result = np.frombuffer(out, dtype=np.uint8, offset=8).reshape(rh, rw, 3)
    return result.copy()  # 脱离 buffer 引用


# ===================================================================
# 方案 B：pypatchmatch C++ 原生
# ===================================================================

def _inpaint_native(
    img: np.ndarray, mask: np.ndarray, patch_size: int
) -> np.ndarray:
    """使用 pypatchmatch C++ 库执行填充。

    Args:
        img (np.ndarray): BGR uint8.
        mask (np.ndarray): 二值掩码 uint8.
        patch_size (int): patch 大小.

    Returns:
        np.ndarray: BGR uint8 结果.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_rgb = _pm_mod.inpaint(rgb, mask, patch_size=patch_size)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


# ===================================================================
# 方案 C：纯 Python 多尺度 PatchMatch（回退）
# ===================================================================

# ---------------------------------------------------------------------------
# 工作区裁剪
# ---------------------------------------------------------------------------

def _crop_work_area(
    img: np.ndarray, mask: np.ndarray, padding: int
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """裁剪到 mask 包围盒 + padding。"""
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img, mask, 0, 0, h, w
    y0 = max(0, int(ys.min()) - padding)
    x0 = max(0, int(xs.min()) - padding)
    y1 = min(h, int(ys.max()) + 1 + padding)
    x1 = min(w, int(xs.max()) + 1 + padding)
    return img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), y0, x0, y1, x1


def _build_pyramid(
    img: np.ndarray, mask: np.ndarray, levels: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """构建高斯图像金字塔（从粗到细）。"""
    pyramid: List[Tuple[np.ndarray, np.ndarray]] = [(img, mask)]
    cur_img, cur_mask = img, mask
    for _ in range(levels - 1):
        cur_img = cv2.pyrDown(cur_img)
        cur_mask = cv2.pyrDown(cur_mask)
        cur_mask = (cur_mask > 127).astype(np.uint8) * 255
        pyramid.append((cur_img, cur_mask))
    pyramid.reverse()
    return pyramid


def _init_nnf(
    h: int, w: int, mask: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """随机初始化 NNF。"""
    nnf = np.zeros((h, w, 2), dtype=np.int32)
    known_ys, known_xs = np.where(mask == 0)
    fill_ys, fill_xs = np.where(mask > 0)
    if len(known_ys) == 0 or len(fill_ys) == 0:
        nnf[:, :, 0] = np.arange(h)[:, None]
        nnf[:, :, 1] = np.arange(w)[None, :]
        return nnf
    idx = rng.randint(0, len(known_ys), size=len(fill_ys))
    nnf[fill_ys, fill_xs, 0] = known_ys[idx]
    nnf[fill_ys, fill_xs, 1] = known_xs[idx]
    nnf[known_ys, known_xs, 0] = known_ys
    nnf[known_ys, known_xs, 1] = known_xs
    return nnf


def _upsample_nnf(
    nnf: np.ndarray, new_h: int, new_w: int, mask: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """上采样 NNF 并修正无效映射。"""
    ch, cw = nnf.shape[:2]
    scale_y, scale_x = new_h / ch, new_w / cw
    new_nnf = np.zeros((new_h, new_w, 2), dtype=np.int32)
    y_up = cv2.resize(nnf[:, :, 0].astype(np.float32), (new_w, new_h),
                       interpolation=cv2.INTER_NEAREST)
    x_up = cv2.resize(nnf[:, :, 1].astype(np.float32), (new_w, new_h),
                       interpolation=cv2.INTER_NEAREST)
    new_nnf[:, :, 0] = np.clip((y_up * scale_y).astype(np.int32), 0, new_h - 1)
    new_nnf[:, :, 1] = np.clip((x_up * scale_x).astype(np.int32), 0, new_w - 1)
    fill_ys, fill_xs = np.where(mask > 0)
    known_ys, known_xs = np.where(mask == 0)
    if len(fill_ys) > 0 and len(known_ys) > 0:
        ty = new_nnf[fill_ys, fill_xs, 0]
        tx = new_nnf[fill_ys, fill_xs, 1]
        bad = mask[ty, tx] > 0
        bad_idx = np.where(bad)[0]
        if len(bad_idx) > 0:
            ri = rng.randint(0, len(known_ys), size=len(bad_idx))
            new_nnf[fill_ys[bad_idx], fill_xs[bad_idx], 0] = known_ys[ri]
            new_nnf[fill_ys[bad_idx], fill_xs[bad_idx], 1] = known_xs[ri]
    return new_nnf


def _compute_ssd_padded(
    filled_pad: np.ndarray, source_pad: np.ndarray, mask_pad: np.ndarray,
    fill_ys: np.ndarray, fill_xs: np.ndarray,
    tgt_ys: np.ndarray, tgt_xs: np.ndarray, half: int,
) -> np.ndarray:
    """预 pad 批量 SSD。"""
    n = len(fill_ys)
    total_ssd = np.zeros(n, dtype=np.float64)
    total_cnt = np.zeros(n, dtype=np.float64)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            qy = tgt_ys + dy
            qx = tgt_xs + dx
            known = mask_pad[qy, qx] == 0
            ki = np.where(known)[0]
            if len(ki) == 0:
                continue
            py = fill_ys[ki] + dy
            px = fill_xs[ki] + dx
            diff = filled_pad[py, px].astype(np.float64) - source_pad[qy[ki], qx[ki]].astype(np.float64)
            ssd = np.einsum("ij,ij->i", diff, diff)
            np.add.at(total_ssd, ki, ssd)
            np.add.at(total_cnt, ki, 1.0)
    total_cnt[total_cnt == 0] = 1e-10
    return total_ssd / total_cnt


def _pm_iteration(
    filled_pad: np.ndarray, source_pad: np.ndarray, mask_pad: np.ndarray,
    mask_orig: np.ndarray, nnf: np.ndarray, nnf_dist: np.ndarray,
    half: int, rng: np.random.RandomState, iteration: int,
):
    """一轮 PatchMatch：传播 + 随机搜索。"""
    h, w = mask_orig.shape[:2]
    fill_ys, fill_xs = np.where(mask_orig > 0)
    n = len(fill_ys)
    if n == 0:
        return
    if iteration % 2 == 1:
        fill_ys = fill_ys[::-1].copy()
        fill_xs = fill_xs[::-1].copy()
    step = 1 if iteration % 2 == 0 else -1
    pfy = fill_ys + half
    pfx = fill_xs + half

    for dy, dx in [(0, step), (step, 0)]:
        nys = fill_ys - dy
        nxs = fill_xs - dx
        valid_nb = (nys >= 0) & (nys < h) & (nxs >= 0) & (nxs < w)
        vi = np.where(valid_nb)[0]
        if len(vi) == 0:
            continue
        cand_y = nnf[nys[vi], nxs[vi], 0] + dy
        cand_x = nnf[nys[vi], nxs[vi], 1] + dx
        in_bounds = (cand_y >= 0) & (cand_y < h) & (cand_x >= 0) & (cand_x < w)
        vi2 = vi[in_bounds]
        cy = cand_y[in_bounds]
        cx = cand_x[in_bounds]
        if len(vi2) == 0:
            continue
        in_known = mask_orig[cy, cx] == 0
        vi3 = vi2[in_known]
        cy = cy[in_known]
        cx = cx[in_known]
        if len(vi3) == 0:
            continue
        new_d = _compute_ssd_padded(
            filled_pad, source_pad, mask_pad,
            pfy[vi3], pfx[vi3], cy + half, cx + half, half,
        )
        old_d = nnf_dist[fill_ys[vi3], fill_xs[vi3]]
        improved = new_d < old_d
        imp = vi3[improved]
        if len(imp) > 0:
            nnf[fill_ys[imp], fill_xs[imp], 0] = cy[improved]
            nnf[fill_ys[imp], fill_xs[imp], 1] = cx[improved]
            nnf_dist[fill_ys[imp], fill_xs[imp]] = new_d[improved]

    search_r = max(h, w) // 2
    while search_r >= 1:
        cur_ty = nnf[fill_ys, fill_xs, 0]
        cur_tx = nnf[fill_ys, fill_xs, 1]
        ry = np.clip(cur_ty + rng.randint(-search_r, search_r + 1, size=n), 0, h - 1)
        rx = np.clip(cur_tx + rng.randint(-search_r, search_r + 1, size=n), 0, w - 1)
        in_known = mask_orig[ry, rx] == 0
        vi = np.where(in_known)[0]
        if len(vi) > 0:
            new_d = _compute_ssd_padded(
                filled_pad, source_pad, mask_pad,
                pfy[vi], pfx[vi], ry[vi] + half, rx[vi] + half, half,
            )
            old_d = nnf_dist[fill_ys[vi], fill_xs[vi]]
            improved = new_d < old_d
            imp = vi[improved]
            if len(imp) > 0:
                nnf[fill_ys[imp], fill_xs[imp], 0] = ry[imp]
                nnf[fill_ys[imp], fill_xs[imp], 1] = rx[imp]
                nnf_dist[fill_ys[imp], fill_xs[imp]] = new_d[improved]
        search_r //= 2


def _vote_padded(
    source_pad: np.ndarray, mask_pad: np.ndarray, mask_orig: np.ndarray,
    nnf: np.ndarray, half: int,
) -> np.ndarray:
    """预 pad 向量化 patch 投票。"""
    h, w = mask_orig.shape[:2]
    c = source_pad.shape[2]
    result_pad = source_pad.copy()
    weight = np.zeros((h + 2 * half, w + 2 * half), dtype=np.float64)
    accum = np.zeros((h + 2 * half, w + 2 * half, c), dtype=np.float64)
    fill_ys, fill_xs = np.where(mask_orig > 0)
    if len(fill_ys) == 0:
        return source_pad[half:half + h, half:half + w].copy()
    tgt_ys = nnf[fill_ys, fill_xs, 0]
    tgt_xs = nnf[fill_ys, fill_xs, 1]
    pfy = fill_ys + half
    pfx = fill_xs + half
    pty = tgt_ys + half
    ptx = tgt_xs + half
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            py = pfy + dy
            px = pfx + dx
            qy = pty + dy
            qx = ptx + dx
            valid = (mask_pad[py, px] > 0) & (mask_pad[qy, qx] == 0)
            vi = np.where(valid)[0]
            if len(vi) == 0:
                continue
            colors = source_pad[qy[vi], qx[vi]].astype(np.float64)
            np.add.at(accum, (py[vi], px[vi]), colors)
            np.add.at(weight, (py[vi], px[vi]), 1.0)
    valid_w = weight > 0
    for ch in range(c):
        result_pad[:, :, ch][valid_w] = (
            accum[:, :, ch][valid_w] / weight[valid_w]
        ).astype(np.float32)
    return result_pad[half:half + h, half:half + w]


def _pad_image(img: np.ndarray, pad: int) -> np.ndarray:
    """反射 padding。"""
    if img.ndim == 2:
        return np.pad(img, pad, mode="reflect")
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def _blend_seam(
    original: np.ndarray, result: np.ndarray, mask: np.ndarray, width: int = 5
) -> np.ndarray:
    """边界羽化融合。"""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    alpha = np.clip(dist / max(width, 1), 0.0, 1.0)
    alpha3 = np.stack([alpha] * 3, axis=-1)
    blended = result.astype(np.float64) * alpha3 + original.astype(np.float64) * (1.0 - alpha3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _inpaint_fallback(
    img: np.ndarray, mask: np.ndarray, patch_size: int,
    em_iters: int = 5, pm_iters: int = 2,
) -> np.ndarray:
    """纯 Python 多尺度 PatchMatch 补全（最终回退）。

    调用方负责 ROI 裁剪/回贴/融合；本函数仅对传入区域执行填充。

    Args:
        img (np.ndarray): BGR uint8（已裁剪的工作区）.
        mask (np.ndarray): 二值掩码 uint8（已裁剪的工作区）.
        patch_size (int): patch 大小.
        em_iters (int, optional): EM 迭代次数, 默认 5.
        pm_iters (int, optional): 每轮 EM 的 PatchMatch 迭代次数, 默认 2.

    Returns:
        np.ndarray: 填充结果, BGR uint8, 与输入同尺寸.
    """
    if patch_size % 2 == 0:
        patch_size += 1
    half = patch_size // 2

    min_dim = min(img.shape[0], img.shape[1])
    pyramid_levels = 1
    while min_dim > 64:
        min_dim //= 2
        pyramid_levels += 1
    pyramid_levels = min(pyramid_levels, 5)

    pyramid = _build_pyramid(img, mask, pyramid_levels)
    rng = np.random.RandomState(42)
    nnf = None
    filled = None
    n_levels = len(pyramid)

    for level_idx, (level_img, level_mask) in enumerate(pyramid):
        lh, lw = level_img.shape[:2]
        if not np.any(level_mask > 0) or not np.any(level_mask == 0):
            continue

        source = level_img.astype(np.float32)
        filled = source.copy()
        rough = cv2.inpaint(level_img, level_mask, 3, cv2.INPAINT_TELEA)
        fill_area = level_mask > 0
        filled[fill_area] = rough[fill_area].astype(np.float32)

        if nnf is None:
            nnf = _init_nnf(lh, lw, level_mask, rng)
        else:
            nnf = _upsample_nnf(nnf, lh, lw, level_mask, rng)

        source_pad = _pad_image(source, half)
        mask_pad = _pad_image(level_mask, half)
        mask_pad[:half, :] = 0
        mask_pad[-half:, :] = 0
        mask_pad[:, :half] = 0
        mask_pad[:, -half:] = 0

        level_em = max(2, em_iters - (n_levels - 1 - level_idx))
        fill_ys, fill_xs = np.where(level_mask > 0)

        nnf_dist = np.full((lh, lw), 1e18, dtype=np.float64)
        if len(fill_ys) > 0:
            filled_pad = _pad_image(filled, half)
            pfy = fill_ys + half
            pfx = fill_xs + half
            nnf_dist[fill_ys, fill_xs] = _compute_ssd_padded(
                filled_pad, source_pad, mask_pad,
                pfy, pfx,
                nnf[fill_ys, fill_xs, 0] + half,
                nnf[fill_ys, fill_xs, 1] + half,
                half,
            )

        for em in range(level_em):
            filled_pad = _pad_image(filled, half)
            for pm in range(pm_iters):
                _pm_iteration(
                    filled_pad, source_pad, mask_pad,
                    level_mask, nnf, nnf_dist,
                    half, rng, em * pm_iters + pm,
                )
            filled = _vote_padded(source_pad, mask_pad, level_mask, nnf, half)

    result = img.copy()
    if filled is not None:
        fill_area = mask > 0
        result[fill_area] = np.clip(filled[fill_area], 0, 255).astype(np.uint8)
    return result
