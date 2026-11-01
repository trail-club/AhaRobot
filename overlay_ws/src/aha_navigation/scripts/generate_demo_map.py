#!/usr/bin/env python3
"""Rasterize the RCJO2026 task area's conservative collision footprints."""

import json
import math
from pathlib import Path


def rasterize(config):
    width, height = config["size"]
    resolution = config["resolution"]
    xmin, ymin, xmax, ymax = config["free_bounds"]
    pixels = bytearray()
    for row in range(height - 1, -1, -1):
        y = config["origin"][1] + (row + 0.5) * resolution
        for col in range(width):
            x = config["origin"][0] + (col + 0.5) * resolution
            value = 254 if xmin < x < xmax and ymin < y < ymax else 205
            for obstacle in config["obstacles"]:
                ox, oy, yaw = obstacle["pose"]
                dx, dy = x - ox, y - oy
                local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
                local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
                # Include partially occupied cells, especially thin walls.
                if (
                    abs(local_x) <= obstacle["size"][0] / 2 + resolution / 2
                    and abs(local_y) <= obstacle["size"][1] / 2 + resolution / 2
                ):
                    value = 0
                    break
            pixels.append(value)
    return f"P5\n{width} {height}\n255\n".encode() + pixels


if __name__ == "__main__":
    maps = Path(__file__).resolve().parents[1] / "maps"
    config = json.loads((maps / "rcjo2026_geometry.json").read_text())
    (maps / "rcjo2026.pgm").write_bytes(rasterize(config))
