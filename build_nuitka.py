"""使用 Nuitka 构建 Material Editor 桌面应用。"""

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
OUTPUT_ROOT = PROJECT_ROOT / "dist-nuitka"
ENTRY_STEM = ENTRY_SCRIPT.stem

OPTIONAL_IMPORTS = (
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

    raise RuntimeError("仅支持在 Windows x86_64 或 macOS Apple Silicon 上构建。")


def _expected_output(system_name: str) -> Path:
    if system_name == "Darwin":
        return OUTPUT_ROOT / f"{APP_NAME}.app"
    return OUTPUT_ROOT / f"{ENTRY_STEM}.dist" / f"{APP_NAME}.exe"


def _clean_previous_build() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)


def _nuitka_command(system_name: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        f"--output-dir={OUTPUT_ROOT}",
        f"--output-filename={APP_NAME}",
    ]

    if system_name == "Windows":
        command.append("--windows-console-mode=disable")
    elif system_name == "Darwin":
        command.extend(
            [
                "--macos-create-app-bundle",
                f"--macos-app-name={APP_NAME}",
            ]
        )

    for module_name in OPTIONAL_IMPORTS:
        command.append(f"--nofollow-import-to={module_name}")

    command.append(str(ENTRY_SCRIPT))
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """使用 Nuitka 构建当前平台对应的桌面应用产物。

    Args:
        argv (Sequence[str] | None): 预留的命令行参数序列，当前不会读取其内容。
    Returns:
        int: 进程退出码。
    Raises:
        RuntimeError: 当前平台或 Python 架构不符合目标构建要求。
        subprocess.CalledProcessError: Nuitka 构建失败。
    """
    del argv

    system_name = _validate_target_platform()
    _clean_previous_build()
    command = _nuitka_command(system_name)

    print(f"[nuitka] 平台: {system_name}")
    print(f"[nuitka] 入口: {ENTRY_SCRIPT}")
    print(f"[nuitka] 预期输出: {_expected_output(system_name)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("[nuitka] 构建完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
