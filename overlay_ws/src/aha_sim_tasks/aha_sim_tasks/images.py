"""Decode ROS image buffers, including padded rows and depth byte order."""

import numpy as np


def image_array(message, *, depth=False):
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if depth:
        if message.encoding != "32FC1":
            raise ValueError("Expected aligned 32FC1 depth in metres")
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        count = 1
    else:
        if message.encoding not in channels:
            raise ValueError(f"Unsupported camera encoding: {message.encoding}")
        dtype, count = np.dtype("u1"), channels[message.encoding]
    row_size = message.width * count * dtype.itemsize
    if message.step < row_size or len(message.data) < message.step * message.height:
        raise ValueError("Invalid camera image dimensions or row stride")
    array = np.ndarray(
        (message.height, message.width, count),
        dtype=dtype,
        buffer=bytes(message.data),
        strides=(message.step, count * dtype.itemsize, dtype.itemsize),
    )
    if depth or count == 1:
        return array[:, :, 0].copy()
    if message.encoding.startswith("bgr"):
        return array[:, :, [2, 1, 0]].copy()
    return array[:, :, :3].copy()
