#!/usr/bin/env python3
"""Start the Mac RealSense (or demo) stream and the ROS/NoVNC camera stack."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VNC_URL = "http://localhost:8080/vnc.html"
COMPOSE = [
    "docker", "compose", "--compatibility", "-p", "aharobot-perception",
    "--profile", "darwin",
    "-f", str(ROOT / "docker/docker-compose.yml"),
    "-f", str(ROOT / "docker/docker-compose.perception-macos.yml"),
    "-f", str(HERE / "docker-compose.run.yml"),
]


def run(command: list[str], *, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, env=env, check=check)


def stream_command(script: Path, env: dict[str, str], elevated: bool, *args: str) -> list[str]:
    command = [sys.executable, str(script), *args]
    if not elevated:
        return command
    preserved = (
        [f"DYLD_LIBRARY_PATH={env['DYLD_LIBRARY_PATH']}"]
        if env.get("DYLD_LIBRARY_PATH") else []
    )
    return ["sudo", "-n", "env", *preserved, *command]


def check_camera(script: Path, env: dict[str, str]) -> bool | None:
    """Return whether the host stream needs sudo, or None on a setup failure."""

    probe = subprocess.run(
        stream_command(script, env, False, "--list-devices"),
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if probe.returncode == 0:
        print(probe.stdout, end="", flush=True)
        return False

    diagnostic = probe.stderr + probe.stdout
    permission_error = any(
        marker in diagnostic
        for marker in ("failed to set power state", "RS2_USB_STATUS_ACCESS")
    )
    if not permission_error or os.geteuid() == 0:
        print(diagnostic, end="", file=sys.stderr)
        return None

    print("[perception] macOS requires administrator access to the RealSense USB device.", flush=True)
    print("[perception] Enter your password for the camera process only.", flush=True)
    if run(["sudo", "-v"], env=env, check=False).returncode:
        return None
    elevated_probe = subprocess.run(
        stream_command(script, env, True, "--list-devices"),
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if elevated_probe.returncode:
        print(elevated_probe.stderr + elevated_probe.stdout, end="", file=sys.stderr)
        return None
    print(elevated_probe.stdout, end="", flush=True)
    return True


def is_reachable(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def is_vnc_ready() -> bool:
    try:
        with urllib.request.urlopen(VNC_URL, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_for_stack(env: dict[str, str], timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        running = subprocess.run(
            COMPOSE + ["ps", "--status", "running", "-q", "aha_project"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if not running:
            raise RuntimeError("ROS container stopped; check the logs above")
        rviz_running = subprocess.run(
            COMPOSE + ["exec", "-T", "aha_project", "pgrep", "-x", "rviz2"],
            cwd=ROOT, env=env, capture_output=True, check=False,
        ).returncode == 0
        if is_reachable(9090) and is_vnc_ready() and rviz_running:
            return
        time.sleep(2)
    raise TimeoutError("rosbridge or NoVNC did not become ready within 120 seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="show a synthetic point cloud without a camera")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the common Jazzy image")
    parser.add_argument("camera_args", nargs=argparse.REMAINDER, help="stream options after --")
    args = parser.parse_args()
    camera_args = args.camera_args[1:] if args.camera_args[:1] == ["--"] else args.camera_args

    if platform.system() != "Darwin":
        parser.error("this launcher is for macOS")
    if args.demo and camera_args:
        parser.error("camera options cannot be used with --demo")

    env = os.environ.copy()
    env.update({
        "USER_ID": str(os.getuid()),
        "GROUP_ID": str(os.getgid()),
        "ROS_DOMAIN_ID": env.get("ROS_DOMAIN_ID", "1"),
        "DISPLAY": "novnc:0.0",
    })
    local_sdk_lib = Path(sys.prefix) / "realsense-sdk/lib"
    legacy_sdk_lib = Path("/private/tmp/aharobot-realsense-install/lib")
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (
            str(local_sdk_lib),
            str(legacy_sdk_lib) if legacy_sdk_lib.is_dir() else "",
            env.get("DYLD_LIBRARY_PATH", ""),
        ))
    )

    stream_script = HERE / ("stream_demo.py" if args.demo else "stream_realsense.py")
    elevated_camera = False
    if not args.demo:
        if importlib.util.find_spec("pyrealsense2") is None:
            managed_venv = Path(sys.prefix).resolve() == (ROOT / ".venv-perception-macos").resolve()
            if not managed_venv:
                print(
                    "[perception] pyrealsense2 is missing from the selected Python environment",
                    file=sys.stderr,
                )
                return 1
            print("[perception] RealSense Python binding is missing; preparing it locally...", flush=True)
            if run([sys.executable, str(HERE / "bootstrap_realsense.py")], env=env, check=False).returncode:
                return 1
        print("[perception] Checking RealSense camera and Python binding...", flush=True)
        elevated_camera = check_camera(stream_script, env)
        if elevated_camera is None:
            return 1
    elif run([sys.executable, "-c", "import numpy, websocket"], env=env, check=False).returncode:
        print("[perception] Install tools/perception/macos/requirements.txt first", file=sys.stderr)
        return 1

    if shutil.which("docker") is None:
        print("[perception] Docker CLI is not installed", file=sys.stderr)
        return 1
    docker_ready = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not docker_ready:
        print("[perception] Starting Docker Desktop...", flush=True)
        if run(["open", "-a", "Docker"], env=env, check=False).returncode:
            print("[perception] Could not start Docker Desktop", file=sys.stderr)
            return 1
        for _ in range(60):
            time.sleep(2)
            docker_ready = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
            if docker_ready:
                break
        if not docker_ready:
            print("[perception] Docker Desktop did not become ready", file=sys.stderr)
            return 1

    existing = subprocess.run(
        COMPOSE + ["ps", "-q"], cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if existing:
        print("[perception] This camera stack is already running; stop it before starting again", file=sys.stderr)
        return 1

    stream: subprocess.Popen | None = None
    stack_started = False
    try:
        image_exists = subprocess.run(
            ["docker", "image", "inspect", "trail/aharobot-jazzy:latest"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if args.rebuild or not image_exists:
            print("[perception] Building the common Jazzy image...", flush=True)
            run([
                "docker", "build", "-t", "trail/aharobot-jazzy:latest",
                "--build-arg", f"USER_ID={env['USER_ID']}",
                "--build-arg", f"GROUP_ID={env['GROUP_ID']}",
                "-f", "docker/Dockerfile", ".",
            ], env=env)

        print("[perception] Starting ROS, rosbridge, RViz and NoVNC...", flush=True)
        stack_started = True
        run(COMPOSE + ["up", "--build", "-d"], env=env)
        print("[perception] Waiting for NoVNC, rosbridge and RViz...", flush=True)
        wait_for_stack(env)

        print("[perception] Starting RGB-D stream...", flush=True)
        stream = subprocess.Popen(
            stream_command(stream_script, env, elevated_camera, *camera_args), cwd=ROOT, env=env,
        )
        time.sleep(2)
        if stream.poll() is not None:
            raise RuntimeError(f"RGB-D stream exited with status {stream.returncode}")

        print(f"\n[perception] Open {VNC_URL} and click Connect.", flush=True)
        print("[perception] RViz will show the RGB image and colored point cloud.", flush=True)
        print("[perception] Press Ctrl-C here to stop the stack.\n", flush=True)
        return stream.wait()
    except KeyboardInterrupt:
        print("\n[perception] Stopping...", flush=True)
        return 130
    except (OSError, RuntimeError, TimeoutError, subprocess.CalledProcessError) as exc:
        print(f"[perception] Startup failed: {exc}", file=sys.stderr)
        if stack_started:
            run(COMPOSE + ["logs", "--tail", "80"], env=env, check=False)
        return 1
    finally:
        if stream is not None and stream.poll() is None:
            stream.terminate()
            try:
                stream.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stream.kill()
                stream.wait()
        if stack_started:
            run(COMPOSE + ["down", "--remove-orphans"], env=env, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
