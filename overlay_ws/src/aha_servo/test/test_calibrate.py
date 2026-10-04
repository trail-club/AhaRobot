"""servo_calibrate.py against the fake bus, with the hand moves simulated."""

import importlib.util
import io
import os
import pty
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from aha_servo.joint_map import load_file
from fake_bus import FakeStsBus

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(PKG, "scripts", "servo_calibrate.py")
HEAD_YAML = os.path.join(PKG, "config", "head.yaml")


def load_script():
    spec = importlib.util.spec_from_file_location("servo_calibrate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CalibrateTest(unittest.TestCase):
    TTY = False  # stdin is a pipe; CalibrateTtyTest uses a pty like a terminal

    def setUp(self):
        self.fake = FakeStsBus({12: 2100, 13: 2000})
        self.tmp = tempfile.mkdtemp()
        self.config = os.path.join(self.tmp, "head.yaml")
        shutil.copy(HEAD_YAML, self.config)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [PKG, env.get("PYTHONPATH")]))
        if self.TTY:
            self.master, stdin = pty.openpty()
        else:
            stdin = subprocess.PIPE
        self.proc = subprocess.Popen(
            [
                sys.executable,
                SCRIPT,
                "--config",
                self.config,
                "--port",
                self.fake.port,
                "--baud",
                "921600",
            ],
            env=env,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if self.TTY:
            os.close(stdin)
        self.out = ""  # only the reader thread appends
        self.seen = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def tearDown(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        if self.TTY:
            os.close(self.master)
        self.fake.close()
        shutil.rmtree(self.tmp)

    def _read(self):
        for char in iter(lambda: self.proc.stdout.read(1), ""):
            self.out += char

    def answer(self, prompt, text="\n", move=None, timeout=10.0):
        """Wait for prompt after the previous one, move a servo by hand, type text."""
        deadline = time.monotonic() + timeout
        while (found := self.out.find(prompt, self.seen)) < 0:
            self.assertLess(
                time.monotonic(), deadline, f"no prompt {prompt!r}:\n{self.out}"
            )
            time.sleep(0.02)
        self.seen = found + len(prompt)
        if move:
            self.fake.move_by_hand(*move)
        if self.TTY:
            os.write(self.master, text.encode())
        else:
            self.proc.stdin.write(text)
            self.proc.stdin.flush()

    def finish(self):
        self.assertEqual(self.proc.wait(timeout=10.0), 0, self.out)
        self.reader.join(timeout=2.0)

    def assert_config_unchanged(self):
        with open(self.config, encoding="utf-8") as f, open(
            HEAD_YAML, encoding="utf-8"
        ) as g:
            self.assertEqual(f.read(), g.read())

    def test_zero_and_sign_written(self):
        self.answer("1) Hold")
        self.assertEqual([self.fake.servo(i).torque for i in (12, 13)], [0, 0])
        # Pan to the right: steps go down (sign -1). Tilt down: steps go up (sign +1).
        self.answer("2) Move joint_head_pan", move=(12, 2100 - 300))
        self.answer("3) Move joint_head_tilt", move=(13, 2000 + 250))
        self.answer("? [y/n]", "y\n")
        self.finish()

        cfg = load_file(self.config)
        pan, tilt = cfg.joint("joint_head_pan"), cfg.joint("joint_head_tilt")
        self.assertEqual((pan.zero, pan.sign), (2100, -1))
        self.assertEqual((tilt.zero, tilt.sign), (2000, 1))
        self.assertEqual([self.fake.servo(i).torque for i in (12, 13)], [0, 0])
        with open(self.config, encoding="utf-8") as f, open(
            HEAD_YAML, encoding="utf-8"
        ) as g:
            self.assertEqual(len(f.read().splitlines()), len(g.read().splitlines()))

    def test_small_move_is_asked_again_and_no_write(self):
        self.answer("1) Hold")
        self.answer("2) Move joint_head_pan", move=(12, 2110))
        self.answer("moved only", "")
        self.answer("2) Move joint_head_pan", move=(12, 2400))
        self.answer("3) Move joint_head_tilt", move=(13, 1700))
        self.answer("? [y/n]", "n\n")
        self.finish()
        self.assertIn("not written", self.out)
        # Hint with the measured values and where they go.
        self.assertIn(self.config, self.out.split("not written", 1)[1])
        self.assertIn("joint_head_pan.zero: 2100", self.out)
        self.assertIn("joint_head_pan.sign: 1", self.out)
        self.assertIn("joint_head_tilt.sign: -1", self.out)
        self.assert_config_unchanged()

    def test_empty_confirmation_is_asked_again(self):
        self.answer("1) Hold")
        self.answer("2) Move joint_head_pan", move=(12, 1800))
        self.answer("3) Move joint_head_tilt", move=(13, 2250))
        self.answer("? [y/n]", "\n")
        self.answer("please type y or n", "")
        self.answer("? [y/n]", "y\n")
        self.finish()
        self.assertEqual(load_file(self.config).joint("joint_head_pan").zero, 2100)

    def test_eof_at_confirmation_is_no(self):
        self.answer("1) Hold")
        self.answer("2) Move joint_head_pan", move=(12, 1800))
        self.answer("3) Move joint_head_tilt", move=(13, 2250))
        self.answer("? [y/n]", "")
        self.proc.stdin.close()
        self.finish()
        self.assertIn("not written", self.out)
        self.assert_config_unchanged()


class CalibrateTtyTest(CalibrateTest):
    TTY = True

    def test_eof_at_confirmation_is_no(self):
        self.skipTest("pipe only")

    def test_extra_enter_is_discarded(self):
        # Double Enter at every step: the extra one must not finish the next step
        # nor answer the write prompt.
        self.answer("1) Hold", "\n\n")
        self.answer("2) Move joint_head_pan", "\n\n", move=(12, 1800))
        self.answer("3) Move joint_head_tilt", "\n\n", move=(13, 2250))
        self.answer("? [y/n]", "y\n")
        self.finish()
        self.assertNotIn("moved only", self.out)
        self.assertNotIn("please type y or n", self.out)
        cfg = load_file(self.config)
        pan, tilt = cfg.joint("joint_head_pan"), cfg.joint("joint_head_tilt")
        self.assertEqual(
            (pan.zero, pan.sign, tilt.zero, tilt.sign), (2100, -1, 2000, 1)
        )


class PromptTest(unittest.TestCase):
    def setUp(self):
        self.calib = load_script()
        self.master, slave = pty.openpty()
        self.stdin = os.fdopen(slave, "r")

    def tearDown(self):
        self.stdin.close()
        os.close(self.master)

    def pending(self):
        return bool(select.select([self.stdin.fileno()], [], [], 0.1)[0])

    def test_flush_drops_typed_ahead_lines(self):
        os.write(self.master, b"\n\n")
        self.assertTrue(self.pending())
        with mock.patch.object(sys, "stdin", self.stdin):
            self.calib.flush_input()
        self.assertFalse(self.pending())

    def test_flush_ignores_non_tty(self):
        with mock.patch.object(sys, "stdin", io.StringIO("y\n")):
            self.calib.flush_input()
            self.assertEqual(sys.stdin.readline(), "y\n")

    def test_confirm_needs_explicit_answer(self):
        with mock.patch.object(
            sys, "stdin", io.StringIO("\nmaybe\nY\n")
        ), redirect_stdout(io.StringIO()) as out:
            self.assertTrue(self.calib.confirm("write?"))
        self.assertEqual(out.getvalue().count("please type y or n"), 2)
        with mock.patch.object(sys, "stdin", io.StringIO("\n")), redirect_stdout(
            io.StringIO()
        ):
            self.assertFalse(self.calib.confirm("write?"))


class DefaultConfigTest(unittest.TestCase):
    def setUp(self):
        self.calib = load_script()

    def test_without_ament_index_uses_source_tree(self):
        with mock.patch.dict(
            sys.modules,
            {"ament_index_python": None, "ament_index_python.packages": None},
        ):
            path = self.calib.default_config()
        self.assertEqual(os.path.realpath(path), os.path.realpath(HEAD_YAML))

    def test_package_not_installed_uses_source_tree(self):
        packages = types.ModuleType("ament_index_python.packages")

        class PackageNotFoundError(KeyError):
            pass

        def get_package_share_directory(name):
            raise PackageNotFoundError(name)

        packages.PackageNotFoundError = PackageNotFoundError
        packages.get_package_share_directory = get_package_share_directory
        with mock.patch.dict(
            sys.modules,
            {
                "ament_index_python": types.ModuleType("ament_index_python"),
                "ament_index_python.packages": packages,
            },
        ):
            path = self.calib.default_config()
        self.assertEqual(os.path.realpath(path), os.path.realpath(HEAD_YAML))

    def test_nothing_found_asks_for_config(self):
        with mock.patch.dict(
            sys.modules,
            {"ament_index_python": None, "ament_index_python.packages": None},
        ), mock.patch.object(
            self.calib, "SOURCE_CONFIG", "/nonexistent/head.yaml"
        ), self.assertRaises(SystemExit) as cm:
            self.calib.default_config()
        self.assertIn("--config", str(cm.exception.code))

    def test_runs_from_source_tree_without_ros(self):
        # Isolated interpreter (no PYTHONPATH, no ROS): finds aha_servo and head.yaml
        # next to the script, then fails only on the (missing) port.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        port = os.path.join(tmp, "no-such-tty")
        code = (
            "import runpy, sys\n"
            "sys.modules['ament_index_python'] = None\n"
            "sys.modules['ament_index_python.packages'] = None\n"
            f"sys.argv = [{SCRIPT!r}, '--port', {port!r}]\n"
            f"runpy.run_path({SCRIPT!r}, run_name='__main__')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn(f"config: {os.path.realpath(HEAD_YAML)}", proc.stdout)
        self.assertIn(f"cannot open {port}", proc.stdout)


if __name__ == "__main__":
    unittest.main()
