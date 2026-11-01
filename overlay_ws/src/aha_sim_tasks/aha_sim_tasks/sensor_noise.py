"""Repeatable policy depth corruption; original ROS messages are never mutated."""

from copy import deepcopy
import numpy as np

from .images import image_array


def noisy_depth(message, model, seed):
    values = image_array(message, depth=True).astype("<f4")
    stamp = message.header.stamp
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, stamp.sec, stamp.nanosec])
    )
    valid = np.isfinite(values) & (values > 0)
    noise = rng.normal(0, model.get("depth_noise_stddev", 0), values.shape)
    values[valid] += noise[valid]
    invalid = rng.random(values.shape) < model.get("depth_dropout", 0)
    values[invalid | ~valid | (values <= 0)] = np.nan
    result = deepcopy(message)
    result.is_bigendian = False
    result.step = result.width * 4
    result.data = values.tobytes()
    return result
