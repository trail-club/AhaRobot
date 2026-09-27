#!/usr/bin/env python3
"""Build the macOS RealSense Python binding into the selected virtualenv."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


SDK_VERSION = "v2.58.4"
SDK_REPOSITORY = "https://github.com/realsenseai/librealsense.git"


def main() -> int:
    if importlib.util.find_spec("pyrealsense2") is not None:
        return 0

    missing = [name for name in ("git", "cmake", "pkg-config", "clang") if not shutil.which(name)]
    if missing:
        print(f"[perception] Missing SDK build tools: {', '.join(missing)}", file=sys.stderr)
        print("[perception] Install them with: brew install cmake libusb pkg-config openssl", file=sys.stderr)
        return 1
    if subprocess.run(["pkg-config", "--exists", "libusb-1.0"], check=False).returncode:
        print("[perception] libusb-1.0 is missing; install it with: brew install libusb", file=sys.stderr)
        return 1

    venv = Path(sys.prefix).resolve()
    source = Path(os.environ.get("PERCEPTION_REALSENSE_SOURCE", venv / "realsense-src"))
    build = venv / "realsense-build"
    install = venv / "realsense-sdk"
    binding_dir = Path(sysconfig.get_path("purelib")) / "pyrealsense2"

    print(f"[perception] Building RealSense SDK {SDK_VERSION} into {venv}", flush=True)
    if not source.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", SDK_VERSION, SDK_REPOSITORY, str(source)],
            check=True,
        )
    elif not (source / "CMakeLists.txt").is_file():
        print(f"[perception] Incomplete SDK source at {source}; remove it and retry", file=sys.stderr)
        return 1

    subprocess.run(
        [
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYTHON_INSTALL_DIR={binding_dir}",
            "-DBUILD_EXAMPLES=OFF", "-DBUILD_GRAPHICAL_EXAMPLES=OFF",
            "-DBUILD_TOOLS=OFF", "-DBUILD_WITH_OPENMP=OFF",
            "-DBUILD_UNIT_TESTS=OFF", "-DFORCE_RSUSB_BACKEND=ON",
            "-DCHECK_FOR_UPDATES=OFF",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    subprocess.run(["cmake", "--install", str(build)], check=True)
    if importlib.util.find_spec("pyrealsense2") is None:
        print("[perception] SDK build finished but pyrealsense2 was not installed", file=sys.stderr)
        return 1
    print("[perception] RealSense Python binding installed in the project venv", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[perception] SDK build failed while running: {exc.cmd}", file=sys.stderr)
        raise SystemExit(exc.returncode)
