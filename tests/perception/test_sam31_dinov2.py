"""Scoring helpers of the SAM 3.1 + DINOv2 scripts (no data, DGX or CUDA)."""

from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "tools/perception/macos/sam31_dinov2"),
)
import common  # noqa: E402
import dino  # noqa: E402
import reference  # noqa: E402

TARGET = [10, 10, 50, 90]
OTHER = [60, 10, 100, 90]


class ScoringTests(unittest.TestCase):
    def test_evaluate_counts_found_wrong_and_missed(self):
        self.assertEqual(common.iou(TARGET, TARGET), 1.0)
        self.assertEqual(common.iou(TARGET, OTHER), 0.0)
        e = common.evaluate([[12, 12, 50, 90], OTHER], [TARGET])
        self.assertEqual((e["found"], e["wrong"], e["missed"]), (1, 1, 0))
        self.assertEqual(e["hit"], [True, False])
        self.assertFalse(e["correct"])
        self.assertTrue(common.evaluate([], [])["correct"])
        self.assertEqual(common.evaluate([], [TARGET])["missed"], 1)

    def test_best_threshold_keeps_the_most_photos_correct(self):
        truth = {"a": [TARGET], "b": [TARGET], "c": []}
        scored = {
            "a": [(0.9, TARGET), (0.4, OTHER)],
            "b": [(0.6, TARGET), (0.5, OTHER)],
            "c": [(0.3, OTHER)],
        }
        # 0.51..0.60 keeps all three correct; the lowest such value is chosen.
        self.assertEqual(common.best_threshold(scored, truth), 0.51)
        self.assertEqual(common.separation(scored, truth), (0.1, 0.6, 0.5))

    def test_summarize_totals(self):
        evaluations = [
            common.evaluate([TARGET], [TARGET]),
            common.evaluate([OTHER], [TARGET]),
        ]
        s = common.summarize(evaluations, [1.0, 3.0])
        self.assertEqual(
            (s["correct"], s["found"], s["wrong"], s["missed"]), (1, 1, 1, 1)
        )
        self.assertEqual((s["mean_seconds"], s["max_seconds"]), (2.0, 3.0))


class PhraseTests(unittest.TestCase):
    def test_detect_uses_the_phrase_saved_at_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory)
            (library / "mug").mkdir()
            reference.save_phrase(library, "mug", "ceramic mug", "vlm")
            # Saved -> no VLM call (no DGX in tests).
            self.assertEqual(
                dino.reference_phrase("mug", None, library=library), "ceramic mug"
            )


class ContainerTests(unittest.TestCase):
    def inspect(self, *outputs):
        results = [
            subprocess.CompletedProcess([], code, out, err)
            for code, out, err in outputs
        ]
        with (
            patch.object(common, "ssh", side_effect=results),
            patch.object(common.time, "sleep"),
        ):
            return common.wait_for_container("c", "u")

    def test_exit_code_once_the_container_stops(self):
        self.assertEqual(
            self.inspect((0, "running 0\n", ""), (0, "exited 1\n", "")), "1"
        )

    def test_a_missing_container_raises_instead_of_waiting_forever(self):
        with self.assertRaisesRegex(RuntimeError, "gone"):
            self.inspect((1, "", "Error: No such object: c"))

    def run_container(self, cleanup_reachable=True):
        """run_in_container whose `docker run` fails; returns (error, cleanup calls)."""
        calls = []

        def ssh(script, **kwargs):
            calls.append((script, kwargs))
            if script.startswith("mktemp"):
                return subprocess.CompletedProcess([], 0, "/tmp/aharobot-exp-abc123\n")
            if script.startswith("docker run"):
                return subprocess.CompletedProcess([], 125, "", "no such image")
            if not cleanup_reachable:
                raise RuntimeError("DGX is not reachable over SSH")
            return subprocess.CompletedProcess([], 0)

        with (
            tempfile.TemporaryDirectory() as work,
            patch.object(common, "ssh", side_effect=ssh),
            patch.object(common, "retry"),
            patch("builtins.print"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                common.run_in_container(work, "dino.py", user="u")
        return caught.exception, [c for c in calls if "docker rm -f" in c[0]]

    def test_docker_run_failure_is_a_runtime_error_and_is_cleaned_up(self):
        error, cleanup = self.run_container()
        self.assertIn("no such image", str(error))
        self.assertEqual(len(cleanup), 1)
        self.assertIn("aharobot-exp-abc123", cleanup[0][0])
        self.assertEqual(cleanup[0][1]["attempts"], 2)

    def test_unreachable_cleanup_keeps_the_original_error(self):
        error, cleanup = self.run_container(cleanup_reachable=False)
        self.assertIn("docker run failed", str(error))
        self.assertEqual(len(cleanup), 1)


if __name__ == "__main__":
    unittest.main()
