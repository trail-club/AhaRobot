"""The installed calibration scripts start (their --help, without ROS; needs
the installed aha_test_frame: colcon test)."""

import os
import subprocess
from pathlib import Path

import pytest
from ament_index_python.packages import get_package_prefix

LIB = Path(get_package_prefix("aha_test_frame")) / "lib" / "aha_test_frame"
SCRIPTS = [
    "arm_cloud_calibrate.py",
    "camera_arm_calibrate.py",
    "camera_plane_calibrate.py",
    "frame_calibrate.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help(script):
    path = LIB / script
    assert os.access(path, os.X_OK), path
    done = subprocess.run(
        [str(path), "--help"], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage:")
