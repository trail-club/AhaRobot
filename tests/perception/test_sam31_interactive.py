"""Interactive transport lifecycle checks without SSH, CUDA or USB hardware."""

import argparse
import base64
from contextlib import redirect_stdout
import importlib.util
from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "overlay_ws/src/aha_perception"))
from aha_perception import sam31_worker  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "sam31_interactive", ROOT / "tools/perception/macos/sam31/interactive.py"
)
interactive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(interactive)


def capture_payload():
    data = BytesIO()
    np.savez_compressed(
        data,
        rgb=np.full((2, 3, 3), 80, dtype=np.uint8),
        depth_m=np.ones((2, 3), dtype=np.float32),
        metadata=json.dumps({"frame": 1}),
    )
    return base64.b64encode(data.getvalue()).decode("ascii")


class WorkerTests(unittest.TestCase):
    def test_model_is_loaded_once_and_prompt_changes_after_request_error(self):
        class Segmenter:
            def __init__(self, prompt, threshold, checkpoint):
                self.prompt, self.threshold = prompt, threshold
                self.prompts = []
                print("model diagnostic")

            def predict(self, rgb):
                self.prompts.append(self.prompt)
                print("inference diagnostic")
                return {
                    "out_binary_masks": np.ones((1, 2, 3), dtype=bool),
                    "out_probs": np.array([0.8], dtype=np.float32),
                }

        model = Segmenter("bottle", 0.5, None)
        factory = Mock(return_value=model)

        def request(prompt):
            return json.dumps({"prompt": prompt, "rgbd": capture_payload()})

        source = StringIO(
            request("bottle")
            + '\n{"prompt":"cup","rgbd":"invalid!"}\n'
            + request("cup")
            + "\n"
        )
        destination = StringIO()
        status = sam31_worker.serve(
            source,
            destination,
            threshold=0.5,
            checkpoint=Path("/models/checkpoint"),
            factory=factory,
        )
        responses = [json.loads(line) for line in destination.getvalue().splitlines()]
        self.assertEqual(status, 0)
        factory.assert_called_once()
        self.assertEqual(
            [r["type"] for r in responses], ["ready", "result", "error", "result"]
        )
        self.assertEqual(model.prompts, ["bottle", "cup"])
        self.assertEqual(responses[3]["summary"]["prompt"], "cup")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            summary = interactive.unpack_result(responses[1], output)
            self.assertEqual(summary["objects"][0]["median_depth_m"], 1)
            self.assertEqual(
                {p.name for p in output.iterdir()}, interactive.RESULT_FILES
            )

    def test_startup_failure_is_protocol_error(self):
        destination = StringIO()
        status = sam31_worker.serve(
            StringIO(),
            destination,
            threshold=0.5,
            checkpoint=Path("missing"),
            factory=Mock(side_effect=RuntimeError("CUDA unavailable")),
        )
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(destination.getvalue())["type"], "error")

    def test_result_archive_rejects_path_traversal(self):
        data = BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("../escape", "bad")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with self.assertRaises(ValueError):
                interactive.unpack_result(
                    {
                        "type": "result",
                        "archive": base64.b64encode(data.getvalue()).decode(),
                    },
                    output,
                )
            self.assertFalse(output.exists())


class SshConnectionTests(unittest.TestCase):
    def test_no_config_destination_prompts_for_roster_user_and_sets_public_key_port(
        self,
    ):
        args = argparse.Namespace(host=None, user=None, port=None, identity_file=None)
        reader = Mock(side_effect=["", "bad user", " taro.yamada "])
        with redirect_stdout(StringIO()):
            interactive.configure_ssh(args, reader)
        command = interactive.ssh_command(args, "true")
        self.assertEqual(reader.call_count, 3)
        self.assertIn("10.99.0.1", command)
        self.assertEqual(command[command.index("-l") + 1], "taro.yamada")
        self.assertEqual(command[command.index("-p") + 1], "2222")

    def test_explicit_user_and_key_do_not_need_interactive_input(self):
        args = argparse.Namespace(
            host=None,
            user="taro.yamada",
            port=None,
            identity_file=Path("/keys/custom key"),
        )
        reader = Mock(side_effect=AssertionError("unexpected username prompt"))
        interactive.configure_ssh(args, reader)
        command = interactive.ssh_command(args, "true")
        self.assertEqual(command[command.index("-i") + 1], "/keys/custom key")
        self.assertIn("IdentitiesOnly=yes", command)
        reader.assert_not_called()

    def test_explicit_alias_keeps_ssh_config_user_and_port(self):
        args = argparse.Namespace(
            host="dgx-spark-vscode", user=None, port=None, identity_file=None
        )
        reader = Mock(side_effect=AssertionError("unexpected username prompt"))
        interactive.configure_ssh(args, reader)
        command = interactive.ssh_command(args, "true")
        self.assertEqual(command[-2:], ["dgx-spark-vscode", "true"])
        self.assertNotIn("-p", command)
        self.assertNotIn("-l", command)
        reader.assert_not_called()

    def test_username_prompt_ctrl_c_exits_before_remote_or_camera_setup(self):
        with patch.dict(interactive.os.environ, {}, clear=True), patch(
            "builtins.input", side_effect=KeyboardInterrupt()
        ), patch.object(interactive, "RemoteWorker") as factory, patch.object(
            interactive.subprocess, "run"
        ) as run, redirect_stdout(StringIO()):
            self.assertEqual(interactive.main([]), 130)
        factory.assert_not_called()
        run.assert_not_called()

    def test_direct_connection_cleanup_retry_contains_user_port_and_key(self):
        args = argparse.Namespace(
            host="10.99.0.1",
            user="taro.yamada",
            port=2222,
            identity_file=Path("/keys/key"),
        )
        worker = interactive.RemoteWorker(args)
        worker.container_requested = True
        errors = StringIO()
        with patch.object(
            worker, "run", side_effect=subprocess.CalledProcessError(255, "ssh")
        ), patch("sys.stderr", errors):
            self.assertFalse(worker.close())
        self.assertIn("-l taro.yamada", errors.getvalue())
        self.assertIn("-p 2222", errors.getvalue())
        self.assertIn("-i /keys/key", errors.getvalue())
        self.assertIn(f"docker rm -f {worker.name}", errors.getvalue())


class InteractionTests(unittest.TestCase):
    def test_enter_reuses_prompt_and_capture_failure_can_retry(self):
        worker = Mock()
        worker.predict.return_value = {"type": "result"}
        args = argparse.Namespace(no_open=True)
        reader = Mock(side_effect=["", "bottle", "", "cup", EOFError()])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                interactive,
                "capture_frame",
                side_effect=[subprocess.CalledProcessError(1, "capture"), None, None],
            ) as capture, patch.object(
                interactive,
                "unpack_result",
                return_value={"objects": [], "inference_seconds": 0.1},
            ), redirect_stdout(StringIO()):
                with self.assertRaises(EOFError):
                    interactive.interact(args, worker, Path(directory), reader)
        self.assertEqual(capture.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in worker.predict.call_args_list], ["bottle", "cup"]
        )

    def test_ctrl_c_always_cleans_up_after_partial_start(self):
        with patch.object(interactive.subprocess, "run"), patch.object(
            interactive, "RemoteWorker"
        ) as factory:
            worker = factory.return_value
            worker.start.side_effect = KeyboardInterrupt()
            worker.close.return_value = True
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(
                StringIO()
            ):
                self.assertEqual(
                    interactive.main(["--host", "dgx-spark", "--output", directory]),
                    130,
                )
            worker.close.assert_called_once()

    def test_cleanup_removes_only_owned_container_after_transport_exit(self):
        worker = interactive.RemoteWorker(argparse.Namespace(host="dgx-spark"))
        worker.process = Mock()
        worker.container_requested = True
        worker.remote_dir = "/tmp/aharobot-sam31-1234"
        with patch.object(worker, "run") as remote, redirect_stdout(StringIO()):
            self.assertTrue(worker.close())
        worker.process.terminate.assert_called_once()
        worker.process.wait.assert_called_once_with(timeout=5)
        self.assertIn(f"docker rm -f {worker.name}", remote.call_args_list[0].args[0])
        self.assertEqual(
            remote.call_args_list[1].args[0], "rm -rf -- /tmp/aharobot-sam31-1234"
        )

    def test_broken_pipe_on_local_close_does_not_skip_remote_stop(self):
        worker = interactive.RemoteWorker(argparse.Namespace(host="dgx-spark"))
        worker.process = Mock()
        worker.process.stdin.close.side_effect = BrokenPipeError("interrupted write")
        worker.container_requested = True
        with patch.object(worker, "run") as remote, redirect_stdout(StringIO()):
            self.assertTrue(worker.close())
        self.assertIn(f"docker rm -f {worker.name}", remote.call_args.args[0])

    def test_remote_stop_failure_reports_failure_and_still_removes_sources(self):
        worker = interactive.RemoteWorker(argparse.Namespace(host="dgx-spark"))
        worker.container_requested = True
        worker.remote_dir = "/tmp/aharobot-sam31-1234"
        with patch.object(
            worker, "run", side_effect=[subprocess.CalledProcessError(255, "ssh"), None]
        ) as remote:
            self.assertFalse(worker.close())
        self.assertEqual(remote.call_count, 2)

    def test_disconnect_and_timeout_stop_the_session(self):
        worker = interactive.RemoteWorker(
            argparse.Namespace(host="dgx-spark", timeout=0.01)
        )
        worker.process = Mock(stdout=StringIO(""))
        worker._read_responses()
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            worker.receive()
        with self.assertRaises(TimeoutError):
            worker.receive()


if __name__ == "__main__":
    unittest.main()
