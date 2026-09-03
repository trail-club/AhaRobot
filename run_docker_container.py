#!/usr/bin/env python3
"""Bring up the AhaRobot dev container and drop the user into a shell.

Supports Linux (native X11), macOS (NoVNC), and WSL2 on Windows (NoVNC).
Auto-detects NVIDIA GPU and layers docker-compose.gpu.yml when present.

Container name is always `aharobot_aha_project_1` regardless of platform.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = "aharobot"
SERVICE = "aha_project"
CONTAINER = f"{PROJECT}_{SERVICE}_1"          # docker compose --compatibility naming
COMPOSE_BASE = "./docker/docker-compose.yml"
COMPOSE_GPU = "./docker/docker-compose.gpu.yml"
COMPOSE_WSL = "./docker/docker-compose.wsl-novnc.yml"
IMAGE = "trail/aharobot-jazzy"


def sh(cmd: str, *, check: bool = True, capture: bool = False, env: dict | None = None):
    merged = None
    if env:
        merged = os.environ.copy()
        merged.update(env)
    return subprocess.run(
        cmd, shell=True, check=check, capture_output=capture, text=True, env=merged
    )


def detect_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def detect_nvidia_gpu() -> bool:
    """Return True when an NVIDIA GPU is reachable via nvidia-smi."""
    candidates = ["nvidia-smi"]
    if platform.system() == "Linux":
        # WSL2 exposes the Windows NVIDIA driver here; not always on PATH.
        candidates.append("/usr/lib/wsl/lib/nvidia-smi")

    for candidate in candidates:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if not found:
            continue
        try:
            r = subprocess.run(
                [found, "-L"], check=False, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0 and "GPU" in r.stdout:
            return True
    return False


def container_exists() -> bool:
    r = sh("docker ps -a --format '{{.Names}}'", check=False, capture=True)
    return bool(r and CONTAINER in (r.stdout or "").split("\n"))


def build_compose_cmd(*, use_novnc: bool, use_gpu: bool, use_wsl_novnc: bool) -> str:
    profile = "darwin" if use_novnc else "linux"
    files = f"-f {COMPOSE_BASE}"
    if use_gpu:
        files += f" -f {COMPOSE_GPU}"
    if use_wsl_novnc:
        files += f" -f {COMPOSE_WSL}"
    return (
        f"docker compose --compatibility -p {PROJECT} "
        f"--profile {profile} {files} up -d"
    )


def start_container(*, display: str, ros_domain_id: int,
                    use_novnc: bool, use_gpu: bool, use_wsl_novnc: bool) -> None:
    env = {
        "DISPLAY": display,
        "ROS_DOMAIN_ID": str(ros_domain_id),
        "USER_ID": str(os.getuid()),
        "GROUP_ID": str(os.getgid()),
    }
    # `up -d` is idempotent: it reconciles config drift (e.g. after toggling
    # GPU) without needing an explicit rm/recreate step.
    cmd = build_compose_cmd(use_novnc=use_novnc, use_gpu=use_gpu, use_wsl_novnc=use_wsl_novnc)
    print(f"$ {cmd}")
    sh(cmd, env=env)


def setup_x11_auth(display: str | None, *, use_novnc: bool, is_wsl: bool) -> None:
    """Copy the host's X11 magic cookie into the container.

    Skipped when GUI goes through NoVNC — the NoVNC container runs its own
    Xvfb so no host cookie is involved.
    """
    if use_novnc or is_wsl:
        return
    if platform.system() != "Linux" or not display:
        return
    try:
        r = sh(f"xauth list {display}", capture=True, check=False)
        entry = (r.stdout or "").strip()
        if not entry:
            sh(f"xauth generate {display} .", check=False)
            r = sh(f"xauth list {display}", capture=True, check=False)
            entry = (r.stdout or "").strip()
        parts = entry.split()
        if len(parts) >= 3:
            proto, key = parts[1], parts[2]
            sh(
                f'docker exec -i {CONTAINER} bash -c '
                f'"touch $HOME/.Xauthority; xauth add {display} {proto} {key}"',
                check=False,
            )
        else:
            print("warn: xauth entry not found; GUI apps may fail.")
    except Exception as e:
        print(f"warn: X11 auth setup failed ({e}); continuing.")


def print_novnc_banner() -> None:
    print()
    print("=" * 60)
    print("GUI apps in the container are shown via NoVNC:")
    print("  1. Open http://localhost:8080/vnc.html in your browser")
    print("  2. Click 'Connect'")
    print("  3. Anything you launch inside the container will appear there")
    print("=" * 60)
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild", action="store_true", help="rebuild the image first")
    p.add_argument(
        "--novnc", action="store_true",
        help="force NoVNC even on native Linux (auto on macOS/WSL)",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="disable NVIDIA GPU passthrough even if detected",
    )
    p.add_argument(
        "--no-enter", action="store_true",
        help="only bring the container up; don't drop into a shell",
    )
    args = p.parse_args()

    os.chdir(Path(__file__).parent)

    system = platform.system()
    is_wsl = detect_wsl()
    is_darwin = system == "Darwin"
    is_linux_native = system == "Linux" and not is_wsl

    # NoVNC decision: mandatory on macOS/WSL, optional on Linux via --novnc.
    use_novnc = is_darwin or is_wsl or args.novnc

    # GPU decision: only when detected and not disabled. NVIDIA passthrough
    # works on Linux native and WSL2. Not applicable on macOS.
    use_gpu = (not args.no_gpu) and (not is_darwin) and detect_nvidia_gpu()

    # WSL-specific compose override: swap host networking for a shared bridge
    # so DISPLAY=novnc:0.0 resolves. Only when actually going through NoVNC.
    use_wsl_novnc = is_wsl and use_novnc

    # DISPLAY resolution:
    #   native Linux X11:  use host's $DISPLAY (usually :0)
    #   macOS  + NoVNC:    :0 (unused — NoVNC has its own Xvfb; base compose
    #                          adds extra_hosts so display:0 also works)
    #   WSL2   + NoVNC:    novnc:0.0 (resolved by compose network)
    if is_wsl and use_novnc:
        display = "novnc:0.0"
    else:
        display = os.environ.get("DISPLAY", ":0") if is_linux_native else ":0"

    print(f"platform : {system}{' (WSL)' if is_wsl else ''}")
    print(f"gpu      : {'NVIDIA' if use_gpu else 'CPU only'}")
    print(f"gui      : {'NoVNC' if use_novnc else 'host X11'}")
    print(f"display  : {display}")
    print(f"container: {CONTAINER}")

    if args.rebuild:
        sh(
            f"docker build -t {IMAGE} "
            f"--build-arg USER_ID={os.getuid()} --build-arg GROUP_ID={os.getgid()} "
            f"-f ./docker/Dockerfile ."
        )

    start_container(
        display=display,
        ros_domain_id=1,
        use_novnc=use_novnc,
        use_gpu=use_gpu,
        use_wsl_novnc=use_wsl_novnc,
    )

    setup_x11_auth(display, use_novnc=use_novnc, is_wsl=is_wsl)

    if use_novnc:
        print_novnc_banner()

    if args.no_enter:
        print(f"container '{CONTAINER}' is up. Attach with:  make shell")
        return 0

    print(f"Entering '{CONTAINER}'. Type 'exit' to leave the shell.")
    sh(f"docker exec -it {CONTAINER} bash", check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
