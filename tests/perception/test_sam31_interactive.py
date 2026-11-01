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

    def test_result_archive_accepts_only_the_optional_candidates_image(self):
        def archive(*extra):
            data = BytesIO()
            with zipfile.ZipFile(data, "w") as result:
                for name in [*interactive.RESULT_FILES, *extra]:
                    content = b"{}" if name == "result.json" else b"x"
                    result.writestr(name, content)
            return {
                "type": "result",
                "archive": base64.b64encode(data.getvalue()).decode(),
                "summary": {},
            }

        with tempfile.TemporaryDirectory() as directory:
            interactive.unpack_result(
                archive("candidates.png"), Path(directory) / "vlm"
            )
            self.assertTrue((Path(directory) / "vlm/candidates.png").is_file())
            interactive.unpack_result(
                archive("candidates.png", "match_3.png"), Path(directory) / "match"
            )
            for name in ("other.png", "match_9.png", "../match_1.png"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    interactive.unpack_result(archive(name), Path(directory) / "x")

    def test_unexpected_request_errors_keep_the_loaded_model(self):
        class Segmenter:
            threshold = 0.5

            def __init__(self):
                self.calls = 0

            def predict(self, rgb):
                self.calls += 1
                if self.calls == 1:
                    raise AssertionError("upstream assertion")
                return {
                    "out_binary_masks": np.ones((1, 2, 3), dtype=bool),
                    "out_probs": np.array([0.8], dtype=np.float32),
                }

        source = StringIO(
            json.dumps({"prompt": 1, "rgbd": capture_payload()})
            + "\n"
            + json.dumps({"prompt": "cup", "rgbd": capture_payload()})
            + "\n"
            + json.dumps({"prompt": "cup", "rgbd": capture_payload()})
            + "\n"
        )
        destination = StringIO()
        status = sam31_worker.serve(
            source,
            destination,
            threshold=0.5,
            checkpoint=Path("/models/checkpoint"),
            factory=Mock(return_value=Segmenter()),
        )
        responses = [json.loads(line) for line in destination.getvalue().splitlines()]
        self.assertEqual(status, 0)
        self.assertEqual(
            [r["type"] for r in responses], ["ready", "error", "error", "result"]
        )
        self.assertIn("string", responses[1]["message"])
        self.assertIn("upstream assertion", responses[2]["message"])

    def test_client_disconnect_does_not_retry_the_failed_write(self):
        destination = Mock()
        destination.write.side_effect = BrokenPipeError("client gone")
        status = sam31_worker.serve(
            StringIO(),
            destination,
            threshold=0.5,
            checkpoint=Path("/models/checkpoint"),
            factory=Mock(),
        )
        self.assertEqual(status, 1)
        destination.write.assert_called_once()

    def test_native_stdout_writes_do_not_reach_the_protocol_channel(self):
        script = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(ROOT / 'overlay_ws/src/aha_perception')!r})\n"
            "from aha_perception.sam31_worker import protocol_stream\n"
            "stream = protocol_stream()\n"
            "os.write(1, b'native diagnostic\\n')\n"
            "print('python diagnostic', flush=True)\n"
            'stream.write(\'{"type": "ready"}\\n\')\n'
            "stream.flush()\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout, '{"type": "ready"}\n')
        self.assertIn("native diagnostic", result.stderr)
        self.assertIn("python diagnostic", result.stderr)


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
        with (
            patch.dict(interactive.os.environ, {}, clear=True),
            patch("builtins.input", side_effect=KeyboardInterrupt()),
            patch.object(interactive, "RemoteWorker") as factory,
            patch.object(interactive.subprocess, "run") as run,
            redirect_stdout(StringIO()),
        ):
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
        with (
            patch.object(
                worker, "run", side_effect=subprocess.CalledProcessError(255, "ssh")
            ),
            patch("sys.stderr", errors),
        ):
            self.assertFalse(worker.close())
        self.assertIn("-l taro.yamada", errors.getvalue())
        self.assertIn("-p 2222", errors.getvalue())
        self.assertIn("-i /keys/key", errors.getvalue())
        self.assertIn(f"docker rm -f {worker.name}", errors.getvalue())


class FlowReportTests(unittest.TestCase):
    def test_flow_report_shows_each_stage_and_escapes_vlm_text(self):
        summary = {
            "prompt": "pringles",
            "inference_seconds": 40.0,
            "objects": [
                {
                    "label": 1,
                    "score": 0.98,
                    "median_depth_m": None,
                    "bbox_xyxy": [1, 2, 3, 4],
                    "color": "#88C26E",
                }
            ],
            "vlm": {
                "model": "qwen3.8-27b",
                "phrases": ["pringles", "red snack can"],
                "candidate_threshold": 0.3,
                "candidates": [
                    {
                        "number": 1,
                        "phrase": "red snack can",
                        "score": 0.98,
                        "kept": True,
                        "color": "#88C26E",
                    },
                    {
                        "number": 2,
                        "phrase": "pringles",
                        "score": 0.74,
                        "kept": False,
                        "color": "#D1859D",
                    },
                ],
                "keep": [1],
                "vlm_seconds": 35.0,
                "steps": [
                    {
                        "name": "phrases",
                        "seconds": 30.0,
                        "answer": '{"phrases": []}',
                        "thinking": "<think><script>x</script>",
                    },
                    {
                        "name": "select",
                        "seconds": 5.0,
                        "answer": '{"keep": [1]}',
                        "thinking": "",
                    },
                ],
                "sam_runs": [
                    {
                        "phrase": "pringles",
                        "seconds": 0.4,
                        "masks": [
                            {
                                "score": 0.74,
                                "bbox_xyxy": [5, 6, 7, 8],
                                "pixels": 9,
                                "outcome": "candidate 2",
                            }
                        ],
                    },
                    {"phrase": "red snack can", "seconds": 0.4, "masks": []},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "candidates.png").write_bytes(b"x")
            page = interactive.write_flow_report(directory, summary).read_text()
        for text in (
            "1. VLMによる言い換え",
            "→ 候補2",
            "検出なし",
            "採用",
            "除外",
            "#D1859D",
            "candidates.png",
        ):
            self.assertIn(text, page)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_flow_report_shows_the_same_product_checks(self):
        candidate = {"phrase": "red snack can", "score": 0.9, "color": "#88C26E"}
        summary = {
            "prompt": "pringles",
            "inference_seconds": 50.0,
            "objects": [],
            "vlm": {
                "model": "qwen3.8-27b",
                "phrases": ["red snack can"],
                "candidate_threshold": 0.3,
                "candidates": [
                    dict(candidate, number=1, kept=True, via="select"),
                    dict(candidate, number=2, kept=False, via=None),
                    dict(candidate, number=3, kept=True, via="match"),
                ],
                "selected": [1],
                "matches": [
                    {"number": 2, "reference": 1, "same": False},
                    {
                        "number": 3,
                        "reference": 1,
                        "same": True,
                        "reference_size": {
                            "height_cm": 23.4,
                            "width_cm": 7.6,
                            "depth_m": 0.61,
                            "cut_off": False,
                        },
                        "size": None,
                    },
                ],
                "keep": [1, 3],
                "vlm_seconds": 40.0,
                "steps": [
                    {"name": "match 3", "seconds": 6.0, "answer": '{"same": true}'},
                ],
                "sam_runs": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            for name in ("candidates.png", "match_3.png"):
                (Path(directory) / name).write_bytes(b"x")
            from PIL import Image

            Image.new("RGB", (400, 300), "white").save(Path(directory) / "rgb.png")
            summary["vlm"]["candidates"][0]["bbox_xyxy"] = [10, 100, 100, 280]
            summary["vlm"]["sam_runs"] = [
                {
                    "phrase": "red snack can",
                    "masks": [
                        {
                            "score": 0.9,
                            "bbox_xyxy": [200, 100, 350, 280],
                            "pixels": 9,
                            "outcome": "candidate 3",
                        }
                    ],
                }
            ]
            page = interactive.write_flow_report(directory, summary).read_text()
            boxes = Image.open(Path(directory) / "match_3_boxes.png")
            # A (candidate 1) and B (candidate 3) are outlined in their colours.
            self.assertEqual(boxes.getpixel((11, 200)), (0x88, 0xC2, 0x6E))
            self.assertEqual(boxes.getpixel((50, 200)), (255, 255, 255))
            self.assertNotEqual(boxes.getpixel((201, 200)), (255, 255, 255))
            self.assertFalse((Path(directory) / "match_2_boxes.png").exists())
        for text in (
            "5. 照合（同じ商品か）",
            "候補1（A） と 候補3（B）",
            "同じ商品 → 採用",
            "別の商品 → 除外",
            "match_3.png",
            "採用（照合）",
            "採用（選択）",
            "最終的な採用: [1, 3]",
            "{&quot;same&quot;: true}",
            "A: 高さ約23 cm・幅約8 cm（距離 0.61 m） / B: 大きさ不明",
            "match_3_boxes.png",
            "SAM 3.1 が検出した枠",
        ):
            self.assertIn(text, page)
        self.assertNotIn("match_2.png", page)


class InteractionTests(unittest.TestCase):
    def test_enter_reuses_prompt_and_capture_failure_can_retry(self):
        worker = Mock()
        worker.predict.return_value = {"type": "result"}
        args = argparse.Namespace(no_open=True)
        reader = Mock(side_effect=["", "bottle", "", "cup", EOFError()])
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    interactive,
                    "capture_frame",
                    side_effect=[
                        subprocess.CalledProcessError(1, "capture"),
                        None,
                        None,
                    ],
                ) as capture,
                patch.object(
                    interactive,
                    "unpack_result",
                    return_value={"objects": [], "inference_seconds": 0.1},
                ),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(EOFError):
                    interactive.interact(args, worker, Path(directory), reader)
        self.assertEqual(capture.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in worker.predict.call_args_list], ["bottle", "cup"]
        )

    def test_damaged_result_is_reported_and_the_session_continues(self):
        worker = Mock()
        worker.predict.side_effect = [
            {"type": "result", "archive": base64.b64encode(b"not a zip").decode()},
            {"type": "result"},
            {"type": "result"},
        ]
        args = argparse.Namespace(no_open=True)
        reader = Mock(side_effect=["bottle", "", "", EOFError()])
        errors = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(interactive, "capture_frame"),
                patch("sys.stderr", errors),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(EOFError):
                    interactive.interact(args, worker, Path(directory), reader)
        self.assertEqual(worker.predict.call_count, 3)
        self.assertEqual(errors.getvalue().count("Result transfer failed"), 3)

    def test_ctrl_c_always_cleans_up_after_partial_start(self):
        with (
            patch.object(interactive.subprocess, "run"),
            patch.object(interactive, "RemoteWorker") as factory,
            patch.object(interactive, "CameraStream") as camera_factory,
        ):
            worker = factory.return_value
            worker.start.side_effect = KeyboardInterrupt()
            worker.close.return_value = True
            camera_factory.return_value.url = "http://127.0.0.1:8765/"
            with (
                tempfile.TemporaryDirectory() as directory,
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    interactive.main(
                        ["--host", "dgx-spark", "--output", directory, "--no-open"]
                    ),
                    130,
                )
            worker.close.assert_called_once()
            camera_factory.return_value.close.assert_called_once()

    def test_live_camera_reports_status_and_capture_errors_can_retry(self):
        worker = Mock()
        worker.predict.return_value = {"type": "result"}
        camera = Mock()
        camera.capture.side_effect = [interactive.CaptureError("no frame"), None]
        args = argparse.Namespace(no_open=True)
        reader = Mock(side_effect=["cup", "", EOFError()])
        errors = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(interactive, "capture_frame") as capture_frame,
                patch.object(
                    interactive,
                    "unpack_result",
                    return_value={"objects": [{}], "inference_seconds": 0.1},
                ),
                patch("sys.stderr", errors),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(EOFError):
                    interactive.interact(
                        args, worker, Path(directory), reader, camera=camera
                    )
        capture_frame.assert_not_called()
        self.assertEqual(camera.capture.call_count, 2)
        self.assertIn("Capture failed: no frame", errors.getvalue())
        camera.show_result.assert_called_once()
        self.assertTrue(
            str(camera.show_result.call_args.args[0]).endswith("result/overlay.png")
        )
        statuses = [call.args[0] for call in camera.set_status.call_args_list]
        self.assertIn("撮影に失敗しました。Enterで再撮影できます", statuses)
        self.assertTrue(statuses[-1].startswith("完了: 「cup」1件"))

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

    def start_worker(self, **options):
        worker = interactive.RemoteWorker(
            argparse.Namespace(
                host="dgx-spark",
                image="local/aharobot-sam31",
                checkpoint="/srv/shared/models/sam3.1/sam3.1_multiplex.pt",
                threshold=0.5,
                **{"vlm": False, "no_vlm_match": False, **options},
            )
        )
        results = [
            Mock(stdout="/tmp/aharobot-sam31-1234\n"),
            None,
            None,
            Mock(stdout="sha256:" + "a" * 64 + "\n"),
            None,
        ]
        with (
            patch.object(worker, "run", side_effect=results),
            patch.object(interactive, "source_archive"),
            patch.object(interactive.subprocess, "Popen") as popen,
            patch.object(interactive.threading, "Thread"),
            patch.object(worker, "receive", return_value={"type": "ready"}),
            redirect_stdout(StringIO()),
        ):
            worker.start()
        return popen.call_args.args[0][-1]

    def test_vlm_mode_uses_host_network_and_passes_vlm_options(self):
        plain = self.start_worker()
        self.assertNotIn("--network", plain)
        self.assertNotIn("--vlm", plain)
        script = self.start_worker(
            vlm=True,
            vlm_url="http://10.99.0.1:8080/v1",
            vlm_model="qwen3.8-27b",
            candidate_threshold=0.3,
        )
        image_id = "sha256:" + "a" * 64
        self.assertIn("--network host " + image_id, script)
        self.assertIn("--vlm --vlm-url http://10.99.0.1:8080/v1", script)
        self.assertIn("--vlm-model qwen3.8-27b --candidate-threshold 0.3", script)
        self.assertNotIn("--no-match", script)
        self.assertIn(
            "--no-match",
            self.start_worker(
                vlm=True,
                vlm_url="http://10.99.0.1:8080/v1",
                vlm_model="qwen3.8-27b",
                candidate_threshold=0.3,
                no_vlm_match=True,
            ),
        )

    def test_container_runs_the_image_id_built_in_this_session(self):
        worker = interactive.RemoteWorker(
            argparse.Namespace(
                host="dgx-spark",
                image="local/aharobot-sam31",
                checkpoint="/srv/shared/models/sam3.1/sam3.1_multiplex.pt",
                threshold=0.5,
                vlm=False,
            )
        )
        image_id = "sha256:" + "a" * 64
        results = [
            Mock(stdout="/tmp/aharobot-sam31-1234\n"),
            None,
            None,
            Mock(stdout=image_id + "\n"),
            None,
        ]
        with (
            patch.object(worker, "run", side_effect=results) as remote,
            patch.object(interactive, "source_archive"),
            patch.object(interactive.subprocess, "Popen") as popen,
            patch.object(interactive.threading, "Thread"),
            patch.object(worker, "receive", return_value={"type": "ready"}),
            redirect_stdout(StringIO()),
        ):
            worker.start()
        self.assertIn("--iidfile", remote.call_args_list[2].args[0])
        script = popen.call_args.args[0][-1]
        self.assertIn(" " + image_id + " ", script)
        self.assertNotIn(" local/aharobot-sam31 ", script)

    def test_non_protocol_output_names_the_offending_line(self):
        worker = interactive.RemoteWorker(
            argparse.Namespace(host="dgx-spark", timeout=0.01)
        )
        worker.process = Mock(stdout=StringIO("Welcome to DGX\n"))
        worker._read_responses()
        with self.assertRaisesRegex(RuntimeError, "Welcome to DGX"):
            worker.receive()

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


FAKE_CAMERA = r"""
import json, sys
from pathlib import Path

def send(message, payload=b""):
    out = sys.stdout.buffer
    out.write(json.dumps(dict(message, bytes=len(payload))).encode() + b"\n" + payload)
    out.flush()

send({"type": "ready", "device": "Fake (serial 1)"})
send({"type": "preview"}, b"JPEG1")
for line in sys.stdin:
    output = Path(json.loads(line)["capture"])
    if output.name == "bad":
        send({"type": "error", "message": "output already exists"})
    else:
        output.mkdir()
        send({"type": "captured", "output": str(output)})
    send({"type": "preview"}, b"JPEG2")
"""


class CameraStreamTests(unittest.TestCase):
    def start_camera(self):
        args = argparse.Namespace(preview_port=0)
        camera = interactive.CameraStream(args)
        self.addCleanup(camera.close)
        with patch.object(
            camera, "command", return_value=[sys.executable, "-c", FAKE_CAMERA]
        ):
            self.assertEqual(camera.start(timeout=10), "Fake (serial 1)")
        return camera

    def test_camera_command_runs_the_serve_mode_as_root_on_macos(self):
        args = argparse.Namespace(
            width=640, height=480, fps=15, warmup_frames=30, serial="123"
        )
        with patch.object(interactive.platform, "system", return_value="Darwin"):
            command = interactive.CameraStream(args).command()
        self.assertEqual(command[:2], ["sudo", "-n"])
        self.assertIn("--serve", command)
        self.assertEqual(command[-2:], ["--serial", "123"])

    def test_capture_requests_get_answers_and_preview_is_served(self):
        from urllib.request import urlopen

        camera = self.start_camera()
        with tempfile.TemporaryDirectory() as directory:
            camera.capture(Path(directory) / "capture")
            self.assertTrue((Path(directory) / "capture").is_dir())
            with self.assertRaisesRegex(interactive.CaptureError, "already exists"):
                camera.capture(Path(directory) / "bad")
        self.assertEqual(camera.frame, b"JPEG2")
        camera.set_status("「cup」を推論中…")
        with urlopen(camera.url + "status", timeout=5) as response:
            self.assertEqual(json.load(response)["status"], "「cup」を推論中…")
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "overlay.png"
            overlay.write_bytes(b"PNG")
            camera.show_result(overlay, "「cup」1件（0001）")
            with urlopen(camera.url + "status", timeout=5) as response:
                status = json.load(response)
            self.assertEqual(
                (status["result"], status["result_label"]), (1, "「cup」1件（0001）")
            )
            with urlopen(camera.url + "result.png?v=1", timeout=5) as response:
                self.assertEqual(response.read(), b"PNG")
        with urlopen(camera.url, timeout=5) as response:
            self.assertIn("/stream", response.read().decode())
        with urlopen(camera.url + "stream", timeout=5) as response:
            self.assertEqual(response.readline(), b"--frame\r\n")
            self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")

    def test_stopped_camera_process_is_a_capture_error(self):
        camera = self.start_camera()
        camera.process.kill()
        camera.process.wait(timeout=10)
        with self.assertRaises(interactive.CaptureError):
            camera.capture(Path("unused"))


class CameraServeTests(unittest.TestCase):
    def test_serve_saves_a_pair_per_request_and_streams_previews(self):
        import queue

        spec = importlib.util.spec_from_file_location(
            "sam31_capture", ROOT / "tools/perception/macos/sam31/capture_realsense.py"
        )
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        depth = np.full((4, 6), 1.0, dtype=np.float32)
        depth[0, 0] = np.nan
        pipeline = Mock()
        args = argparse.Namespace(width=6, height=4, fps=15, warmup_frames=2)
        out = BytesIO()
        with tempfile.TemporaryDirectory() as directory:
            requests = queue.Queue()
            for line in (
                json.dumps({"capture": str(Path(directory) / "0001/capture")}),
                json.dumps({"capture": directory}),
                None,
            ):
                requests.put(line)
            with (
                patch.object(capture, "_load_realsense"),
                patch.object(
                    capture, "start_pipeline", return_value=(pipeline, Mock())
                ),
                patch.object(
                    capture, "read_pair", return_value=(rgb, depth, {"frame": 1})
                ),
                patch.object(capture, "_device_label", return_value="Fake"),
            ):
                self.assertEqual(capture.serve(args, requests, out), 0)
            saved = np.load(Path(directory) / "0001/capture/rgbd.npz")
            self.assertEqual(saved["rgb"].shape, (4, 6, 3))
        pipeline.stop.assert_called_once()
        stream = BytesIO(out.getvalue())
        messages = []
        while line := stream.readline():
            message = json.loads(line)
            payload = stream.read(message["bytes"])
            messages.append((message["type"], payload))
        self.assertEqual(
            [kind for kind, _ in messages],
            ["ready", "captured", "preview", "error", "preview"],
        )
        self.assertTrue(messages[2][1].startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
