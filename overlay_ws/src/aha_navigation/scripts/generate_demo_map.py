#!/usr/bin/env python3
"""Rasterize the RCJO2026 task area's conservative collision footprints."""

import json
import math
from pathlib import Path


def rasterize(config):
    width, height = config["size"]
    resolution = config["resolution"]
    # Cell centres count from cell_anchor, a cell corner, so that the cells of
    # the arena do not change when the grid grows around it.
    anchor = config.get("cell_anchor", config["origin"])
    shift = [round((o - a) / resolution) for o, a in zip(config["origin"], anchor)]
    # The arena, and the areas outside it that the robot uses (the entrance).
    free = [config["free_bounds"]] + [a["bounds"] for a in config.get("free_areas", [])]
    pixels = bytearray()
    for row in range(height - 1, -1, -1):
        y = anchor[1] + (row + shift[1] + 0.5) * resolution
        for col in range(width):
            x = anchor[0] + (col + shift[0] + 0.5) * resolution
            inside = any(x0 < x < x1 and y0 < y < y1 for x0, y0, x1, y1 in free)
            value = 254 if inside else 205
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
