#!/usr/bin/env python3
"""Test only: aha_description's controllers.yaml with the test bed overrides.

Mappings are merged key by key, other values of the overrides replace the
base ones, so the test bed keeps up with the robot's controller config.

Usage:  merge_controllers.py <base.yaml> <overrides.yaml> <output.yaml>
"""

import sys
from pathlib import Path

import yaml


def merge(base, overrides):
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides
    out = dict(base)
    for key, value in overrides.items():
        out[key] = merge(base[key], value) if key in base else value
    return out


def main(base: str, overrides: str, dst: str) -> int:
    merged = merge(
        yaml.safe_load(Path(base).read_text()),
        yaml.safe_load(Path(overrides).read_text()),
    )
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    header = f"# Generated from {Path(base).name} and {Path(overrides).name}.\n"
    Path(dst).write_text(header + yaml.safe_dump(merged, sort_keys=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
