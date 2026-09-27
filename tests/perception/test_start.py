import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).parents[2] / "tools/perception/macos/start.py"
SPEC = importlib.util.spec_from_file_location("perception_start", SCRIPT)
start = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(start)


class CameraPrivilegeTests(unittest.TestCase):
    def test_usb_access_error_retries_only_camera_probe_with_sudo(self):
        env = {"DYLD_LIBRARY_PATH": "/private/tmp/rs/lib"}
        denied = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="",
            stderr="failed to set power state: RS2_USB_STATUS_ACCESS",
        )
        allowed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="RealSense D435I\n", stderr="",
        )
        with (
            patch.object(start.os, "geteuid", return_value=501),
            patch.object(start.subprocess, "run", side_effect=[denied, allowed]) as probe,
            patch.object(start, "run", return_value=subprocess.CompletedProcess([], 0)) as sudo,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            elevated = start.check_camera(SCRIPT, env)

        self.assertTrue(elevated)
        self.assertEqual(sudo.call_args.args[0], ["sudo", "-v"])
        elevated_command = probe.call_args_list[1].args[0]
        self.assertEqual(elevated_command[:3], ["sudo", "-n", "env"])
        self.assertIn("DYLD_LIBRARY_PATH=/private/tmp/rs/lib", elevated_command)
        self.assertIn(str(SCRIPT), elevated_command)
        self.assertEqual(elevated_command[-1], "--list-devices")

    def test_other_device_errors_do_not_request_sudo(self):
        missing = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="no RealSense device found",
        )
        with (
            patch.object(start.subprocess, "run", return_value=missing),
            patch.object(start, "run") as sudo,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            elevated = start.check_camera(SCRIPT, {})

        self.assertIsNone(elevated)
        sudo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
