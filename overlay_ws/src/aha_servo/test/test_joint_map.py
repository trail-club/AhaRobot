"""Joint mapping: steps <-> rad, config parsing and yaml write-back."""

import math
import os
import unittest

import yaml

from aha_servo.joint_map import (
    JointConfig,
    load_file,
    parse_config,
    ros_parameters,
    update_yaml_text,
    wrap_steps,
)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD_YAML = os.path.join(PKG, "config", "head.yaml")
PERCEPTION_HEAD_YAML = os.path.join(PKG, "..", "aha_perception", "config", "head.yaml")


class ConversionTests(unittest.TestCase):
    def test_upstream_convention(self):
        # upstream head_controller.py: rad = -(raw - 2048) / 4096 * 2 pi
        j = JointConfig("j", 1, zero=2048, sign=-1)
        self.assertAlmostEqual(j.to_rad(2048), 0.0)
        self.assertAlmostEqual(j.to_rad(1024), math.pi / 2)
        self.assertAlmostEqual(j.to_rad(3072), -math.pi / 2)
        self.assertEqual(j.to_steps(math.pi / 2), 1024)
        self.assertEqual(j.to_steps(-0.5), 2048 + round(0.5 / (2 * math.pi) * 4096))

    def test_zero_and_sign(self):
        j = JointConfig("j", 1, zero=1000, sign=1)
        self.assertEqual(j.to_steps(0.0), 1000)
        self.assertAlmostEqual(j.to_rad(2024), math.pi / 2)
        for steps in (0, 517, 1000, 4095):
            self.assertEqual(j.to_steps(j.to_rad(steps)), steps)

    def test_speed(self):
        j = JointConfig("j", 1, sign=-1)
        self.assertAlmostEqual(j.speed_to_steps(-2 * math.pi), 4096)
        self.assertAlmostEqual(j.speed_to_rad(4096), -2 * math.pi)

    def test_clamp_and_range(self):
        j = JointConfig("j", 1, min=-0.5, max=1.0)
        self.assertEqual(j.clamp(2.0), (1.0, True))
        self.assertEqual(j.clamp(-1.0), (-0.5, True))
        self.assertEqual(j.clamp(0.2), (0.2, False))
        lo, hi = j.step_range()
        self.assertEqual((lo, hi), (j.to_steps(1.0), j.to_steps(-0.5)))

    def test_wrap_steps(self):
        self.assertEqual(wrap_steps(100), 100)
        self.assertEqual(wrap_steps(4000), -96)
        self.assertEqual(wrap_steps(-4000), 96)


class ConfigTests(unittest.TestCase):
    def params(self, bus=None, **joint):
        block = {"id": 12, **joint}
        return {"controller_name": "c", "joints": ["a"], "a": block, **(bus or {})}

    def test_defaults(self):
        cfg = parse_config(self.params())
        self.assertEqual(cfg.baud, 115200)
        j = cfg.joint("a")
        self.assertEqual((j.id, j.zero, j.sign, j.steps_per_rev), (12, 2048, -1, 4096))

    def test_errors(self):
        for bad in (
            {"joints": []},
            {"joints": ["a"]},
            {"joints": ["a"], "a": {"zero": 1}},
            self.params(sign=0),
            self.params(sign=2),
            self.params(min=1.0, max=0.0),
            self.params(min=float("nan")),
            {"joints": ["a", "b"], "a": {"id": 1}, "b": {"id": 1}},
            self.params(id=-1),
            self.params(id=254),
            self.params(zero=-1),
            self.params(zero=4096),
            self.params(zero=1024, steps_per_rev=1024),
            self.params(steps_per_rev=0),
            self.params({"torque_limit": 70000}),
            self.params({"torque_limit": -1}),
            self.params({"acc": 255}),
            self.params({"acc": -1}),
            self.params({"baud": 12345}),
            self.params({"rate_hz": 0.0}),
            self.params({"max_speed": -1.0}),
            self.params({"timeout": 0.0}),
            self.params({"goal_tolerance": 0.0}),
            self.params({"goal_timeout": float("inf")}),
            self.params({"state_timeout": float("nan")}),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_config(bad)

    def test_range_limits_accepted(self):
        for bus, joint in (
            ({"torque_limit": 0, "acc": 0, "baud": 1000000}, {"id": 0, "zero": 0}),
            ({"torque_limit": 1000, "acc": 254, "baud": 921600}, {"id": 253}),
            ({}, {"zero": 4095}),
            ({}, {"zero": 1023, "steps_per_rev": 1024, "sign": 1}),
        ):
            with self.subTest(bus=bus, joint=joint):
                parse_config(self.params(bus, **joint))

    def test_error_names_the_key(self):
        with self.assertRaisesRegex(ValueError, "torque_limit 70000 .* 0..1000"):
            parse_config(self.params({"torque_limit": 70000}))
        with self.assertRaisesRegex(ValueError, r"a\.zero 4096 .* 0..4095"):
            parse_config(self.params(zero=4096))

    def test_head_yaml(self):
        cfg = load_file(HEAD_YAML)
        self.assertEqual(cfg.controller_name, "head_controller")
        self.assertEqual(cfg.joint_names, ["joint_head_pan", "joint_head_tilt"])
        self.assertEqual([j.id for j in cfg.joints], [12, 13])

    @unittest.skipUnless(
        os.path.exists(PERCEPTION_HEAD_YAML), "aha_perception not next to aha_servo"
    )
    def test_head_limits_match_aha_perception(self):
        cfg = load_file(HEAD_YAML)
        with open(PERCEPTION_HEAD_YAML, encoding="utf-8") as f:
            head = ros_parameters(yaml.safe_load(f))["head"]
        pan, tilt = cfg.joint("joint_head_pan"), cfg.joint("joint_head_tilt")
        self.assertEqual([pan.min, pan.max], head["pan_limits"])
        self.assertEqual([tilt.min, tilt.max], head["tilt_limits"])
        self.assertEqual(cfg.max_speed, head["max_speed"])


class YamlWriteBackTests(unittest.TestCase):
    def test_update_keeps_comments_and_other_joints(self):
        with open(HEAD_YAML, encoding="utf-8") as f:
            text = f.read()
        before = parse_config(ros_parameters(yaml.safe_load(text))).joint(
            "joint_head_pan"
        )
        out = update_yaml_text(text, "joint_head_tilt", {"zero": 1990, "sign": 1})
        cfg = parse_config(ros_parameters(yaml.safe_load(out)))
        self.assertEqual(cfg.joint("joint_head_tilt").zero, 1990)
        self.assertEqual(cfg.joint("joint_head_tilt").sign, 1)
        self.assertEqual(cfg.joint("joint_head_pan").zero, before.zero)
        self.assertEqual(cfg.joint("joint_head_pan").sign, before.sign)
        old = [line for line in text.splitlines() if line.lstrip().startswith("#")]
        new = [line for line in out.splitlines() if line.lstrip().startswith("#")]
        self.assertEqual(old, new)
        self.assertEqual(len(out.splitlines()), len(text.splitlines()))

    def test_inline_comment_and_missing_key(self):
        text = "x:\n  ros__parameters:\n    j:\n      id: 3\n      zero: 10  # note\n    k:\n      id: 4\n"
        out = update_yaml_text(text, "j", {"zero": 20, "sign": 1})
        self.assertIn("      zero: 20  # note\n", out)
        self.assertIn("      sign: 1\n", out)
        self.assertTrue(out.endswith("    k:\n      id: 4\n"))
        with self.assertRaises(ValueError):
            update_yaml_text(text, "missing", {"zero": 1})


if __name__ == "__main__":
    unittest.main()
