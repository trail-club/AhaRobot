"""GPSR person phrases and matching against ClassifyPerson answers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.person_description import Description, parse, parse_phrase  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_command_generator_phrases(self):
        cases = {
            "waving person": Description(gesture="waving"),
            "waving persons": Description(gesture="waving"),
            "person raising their left arm": Description(gesture="raising_left_arm"),
            "persons raising their right arm": Description(gesture="raising_right_arm"),
            "person pointing to the left": Description(gesture="pointing_left"),
            "persons pointing to the right": Description(gesture="pointing_right"),
            "sitting person": Description(posture="sitting"),
            "standing persons": Description(posture="standing"),
            "lying person": Description(posture="lying"),
            "person wearing a red t shirt": Description(clothing=("red", "t shirt")),
            "person wearing an orange jacket": Description(
                clothing=("orange", "jacket")
            ),
            "people wearing gray sweaters": Description(clothing=("gray", "sweater")),
            "raising_left_arm": Description(gesture="raising_left_arm"),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase):
                self.assertEqual(parse_phrase(phrase), expected)

    def test_unknown_phrases_are_rejected(self):
        for phrase in ("", "person", "person wearing a red hat", "dancing person"):
            with self.subTest(phrase):
                with self.assertRaises(ValueError):
                    parse_phrase(phrase)

    def test_params_combine_and_contradictions_fail(self):
        description = parse(
            {"pose": "sitting person", "clothing": "person wearing a blue coat"}
        )
        self.assertEqual(
            description, Description(posture="sitting", clothing=("blue", "coat"))
        )
        self.assertEqual(parse({"name": " Jules "}).name, "Jules")
        with self.assertRaises(ValueError):
            parse(
                {
                    "gesture": "waving person",
                    "description": "person pointing to the left",
                }
            )
        with self.assertRaises(ValueError):
            parse({"gesture": 3})
        self.assertTrue(parse({}).empty)


class MatchTests(unittest.TestCase):
    def test_gesture_posture_and_clothing(self):
        waving = Description(gesture="waving")
        self.assertTrue(waving.matches(gesture="waving"))
        self.assertFalse(waving.matches(gesture="none"))
        self.assertIsNone(waving.matches(gesture=""))
        left = Description(gesture="raising_left_arm")
        self.assertTrue(left.matches(gesture="raising_both_arms"))
        red = Description(clothing=("red", "t shirt"))
        self.assertTrue(red.matches(garments=["black jacket", "red t shirt"]))
        self.assertFalse(red.matches(garments=["red jacket"]))
        self.assertIsNone(red.matches(garments=[]))
        self.assertTrue(red.needs_vlm)

    def test_all_parts_must_match(self):
        description = Description(posture="sitting", clothing=("blue", "coat"))
        self.assertTrue(description.matches(posture="sitting", garments=["blue coat"]))
        self.assertFalse(description.matches(posture="standing", garments=[]))
        self.assertIsNone(description.matches(posture="sitting", garments=[]))
        named = Description(name="Jules")
        self.assertTrue(named.matches(name="Jules"))
        self.assertIsNone(named.matches())


if __name__ == "__main__":
    unittest.main()
