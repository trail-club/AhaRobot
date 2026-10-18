"""Scoring helpers of the SAM 3.1 + DINOv2 scripts (no data, DGX or CUDA)."""

from pathlib import Path
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
