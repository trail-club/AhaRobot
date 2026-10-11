"""Conservative grid reachability and reproducible map generation."""

from collections import deque
import importlib.util
import json
import math
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def test_map_reproducible_and_all_task_poses_reachable():
    spec = importlib.util.spec_from_file_location(
        "generate_demo_map", PACKAGE / "scripts" / "generate_demo_map.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((PACKAGE / "maps" / "rcjo2026_geometry.json").read_text())
    generated = module.rasterize(config)
    assert generated == (PACKAGE / "maps" / "rcjo2026.pgm").read_bytes()
    width, height = config["size"]
    pixels = generated.split(b"\n", 3)[3]
    resolution = config["resolution"]
    # Match the configured robot radius, plus one grid cell of clearance.
    radius = 0.45
    cells = math.ceil(radius / resolution)
    offsets = [
        (dx, dy)
        for dx in range(-cells, cells + 1)
        for dy in range(-cells, cells + 1)
        if math.hypot(dx, dy) * resolution <= radius
    ]
    free = set()
    for row in range(cells, height - cells):
        for col in range(cells, width - cells):
            if all(pixels[(row + dy) * width + col + dx] == 254 for dx, dy in offsets):
                free.add((col, row))

    def cell(pose):
        return (
            int((pose[0] - config["origin"][0]) / resolution),
            height - 1 - int((pose[1] - config["origin"][1]) / resolution),
        )

    start = cell([-2, 1.5])
    assert start in free
    visited, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for candidate in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if candidate in free and candidate not in visited:
                visited.add(candidate)
                queue.append(candidate)
    flow = json.loads(
        (PACKAGE.parent / "aha_tasks" / "config" / "human_interaction.json").read_text()
    )
    for label, pose in flow["locations"].items():
        assert cell(pose) in visited, f"{label} is blocked or disconnected"
