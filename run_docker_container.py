#!/usr/bin/env python3
"""Bring up the AhaRobot dev container and drop the user into a shell.

Adapted from kachaka_challenge_trail2026/run_docker_container.py:
  - Fixed compose project name (aharobot)
  - Adds macOS NoVNC guidance
  - No real-robot bridge (added later when hardware is wired)
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT = "aharobot"
CONTAINER = f"{PROJECT}-aha_project-1"
COMPOSE_FILE = "./docker/docker-compose.yml"


def sh(cmd: str, *, check: bool = True, capture: bool = False, env: dict | None = None):
    merged_env = None
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)
    return subprocess.run(
        cmd, shell=True, check=check, capture_output=capture, text=True, env=merged_env
    )


def container_exists() -> bool:
    r = sh("docker ps -a --format '{{.Names}}'", check=False, capture=True)
    return bool(r and CONTAINER in (r.stdout or "").split("\n"))


def start_container(display: str, profile: str, ros_domain_id: int) -> None:
    env = {
        "DISPLAY": display,
        "ROS_DOMAIN_ID": str(ros_domain_id),
        "USER_ID": str(os.getuid()),
        "GROUP_ID": str(os.getgid()),
    }
    if container_exists():
        print(f"Container '{CONTAINER}' exists — starting.")
        try:
            sh(f"docker start {CONTAINER}")
            return
        except subprocess.CalledProcessError:
            print("Start failed; recreating.")
            sh(f"docker rm -f {CONTAINER}", check=False)

    cmd = (
        f"docker compose --compatibility -p {PROJECT} "
        f"--profile {profile} -f {COMPOSE_FILE} up -d"
    )
    sh(cmd, env=env)


def setup_x11_auth(display: str | None) -> None:
    if platform.system() == "Darwin":
        print("macOS: use NoVNC (http://localhost:8080/vnc.html) for GUI apps.")
        return
    if not display:
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
                f'docker exec -it {CONTAINER} bash -c '
                f'"touch $HOME/.Xauthority; xauth add {display} {proto} {key}"',
                check=False,
            )
    except Exception as e:
        print(f"X11 auth setup skipped: {e}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="rebuild the image first")
    args = p.parse_args()

    os.chdir(Path(__file__).parent)

    if args.rebuild:
        sh(
            f"docker build -t trail/aharobot-jazzy "
            f"--build-arg USER_ID={os.getuid()} --build-arg GROUP_ID={os.getgid()} "
            f"-f ./docker/Dockerfile ."
        )

    display = os.environ.get("DISPLAY", ":0")
    profile = "linux"
    if platform.system() == "Darwin":
        display = ":0"
        profile = "darwin"
        print(
            "\n" + "=" * 60 + "\n"
            "macOS detected. To see GUI apps (RViz/Gazebo) from the container:\n"
            "  1. Open http://localhost:8080/vnc.html\n"
            "  2. Click 'Connect'\n"
            + "=" * 60 + "\n"
        )

    start_container(display, profile, ros_domain_id=1)
    setup_x11_auth(display)
    print(f"Entering '{CONTAINER}'. Type 'exit' to leave the shell.")
    sh(f"docker exec -it {CONTAINER} bash", check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
