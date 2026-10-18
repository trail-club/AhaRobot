"""Depth noise is seeded, preserves calibration/stamps, and leaves source data intact."""

import numpy as np
import pytest

pytest.importorskip("rclpy")
from sensor_msgs.msg import Image  # noqa: E402
from aha_sim_tasks.images import image_array  # noqa: E402
from aha_sim_tasks.sensor_noise import noisy_depth  # noqa: E402


def test_depth_noise_is_repeatable_and_does_not_change_original_message():
    pixels = np.full((80, 100), 1.0, dtype="<f4")
    pixels[0, 0] = np.nan
    message = Image(
        height=80, width=100, encoding="32FC1", step=400, data=pixels.tobytes()
    )
    message.header.stamp.sec = 12
    model = {"depth_noise_stddev": 0.005, "depth_dropout": 0.02}
    a = noisy_depth(message, model, 42)
    b = noisy_depth(message, model, 42)
    c = noisy_depth(message, model, 43)
    assert bytes(a.data) == bytes(b.data)
    assert bytes(a.data) != bytes(c.data)
    assert a.header == message.header
    assert bytes(message.data) == pixels.tobytes()
    result = image_array(a, depth=True)
    assert np.isnan(result[0, 0])
    assert 0.01 < np.isnan(result).mean() < 0.03
    assert 0.0045 < np.nanstd(result) < 0.0055


def test_ideal_profile_preserves_all_valid_depths():
    pixels = np.array([[0, 0.5, 1, 2]], dtype="<f4")
    message = Image(height=1, width=4, encoding="32FC1", step=16, data=pixels.tobytes())
    result = image_array(noisy_depth(message, {}, 0), depth=True)
    assert np.isnan(result[0, 0])
    np.testing.assert_array_equal(result[0, 1:], pixels[0, 1:])
