"""构建 Material Editor 桌面应用。"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


APP_NAME = "MaterialEditor"
PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_SCRIPT = PROJECT_ROOT / "main.py"
BUILD_ROOT = PROJECT_ROOT / "build"
DIST_ROOT = PROJECT_ROOT / "dist"

OPTIONAL_EXCLUDES = (
    "torch",
    "torchvision",
    "simple_lama_inpainting",
    "libcom",
    "einops",
    "patchmatch",
    "pypatchmatch",
)


def _validate_target_platform() -> str:
    system_name = platform.system()
    if system_name == "Windows":
        if sys.maxsize <= 2**32:
            raise RuntimeError("Windows 目标需要使用 64 位 Python 构建。")
        return system_name

    if system_name == "Darwin":
        machine = platform.machine().lower()
        if machine not in {"arm64", "aarch64"}:
            raise RuntimeError("macOS 目标需要在 Apple 芯片 Mac 上构建。")
        return system_name

    if system_name == "Linux":
        machine = platform.machine().lower()
        if machine not in {"x86_64", "amd64"}:
            raise RuntimeError("Linux 目标需要使用 x86_64 环境构建。")
        return system_name

    raise RuntimeError(
        "仅支持在 Windows x86_64、Linux x86_64 或 macOS Apple Silicon 上构建。"
    )


def _dist_target(system_name: str) -> Path:
    if system_name == "Darwin":
        return DIST_ROOT / f"{APP_NAME}.app"
    return DIST_ROOT / APP_NAME


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _clean_previous_build(system_name: str) -> None:
    _remove_path(BUILD_ROOT / APP_NAME)
    _remove_path(BUILD_ROOT / f"{APP_NAME}.spec")
    _remove_path(_dist_target(system_name))


def _pyinstaller_command(system_name: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APP_NAME,
        "--workpath",
        str(BUILD_ROOT),
        "--specpath",
        str(BUILD_ROOT),
        "--distpath",
        str(DIST_ROOT),
    ]

    if system_name == "Darwin":
        command.extend(["--target-arch", "arm64"])

    for module_name in OPTIONAL_EXCLUDES:
        command.extend(["--exclude-module", module_name])

    command.append(str(ENTRY_SCRIPT))
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """构建当前平台对应的桌面应用产物。

    Args:
        argv (Sequence[str] | None): 预留的命令行参数序列，当前不会读取其内容。

    Returns:
        int: 进程退出码。

    Raises:
        RuntimeError: 当前平台或 Python 架构不符合目标构建要求。
        subprocess.CalledProcessError: PyInstaller 构建失败。
    """
    del argv

    system_name = _validate_target_platform()
    _clean_previous_build(system_name)
    command = _pyinstaller_command(system_name)

    print(f"[build] 平台: {system_name}")
    print(f"[build] 入口: {ENTRY_SCRIPT}")
    print(f"[build] 输出: {_dist_target(system_name)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("[build] 构建完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
