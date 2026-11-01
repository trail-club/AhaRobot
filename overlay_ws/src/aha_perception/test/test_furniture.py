import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aha_perception.furniture import (
    AssociationOptions,
    FurnitureConfigError,
    associate_surfaces,
    clip_plane,
    level_markers,
    load_config,
    match_point,
    missing_message,
    parse_config,
    requested_surfaces,
    resolve_request,
)
from aha_perception.place import PlaceStatus
from aha_perception.planes.segmentation import make_plane

ARENA = Path(__file__).resolve().parents[1] / "config" / "furniture"


def config_data(**cabinet):
    item = {
        "name": "cabinet",
        "category": "shelf",
        "pose": {"x": 1.2, "y": -0.4, "yaw": math.pi / 2},
        "size": {"x": 0.35, "y": 0.80},
        "levels": [0.05, 0.42, 0.78, 1.12],
        "board_thickness": 0.02,
    }
    item.update(cabinet)
    table = {
        "name": "table",
        "pose": {"x": -1.0, "y": 0.0},
        "size": {"x": 1.2, "y": 0.8},
        "levels": [0.75],
    }
    return {"frame": "map", "furniture": [item, table]}


def surface(sid, centroid, label="shelf", vertical=False):
    return SimpleNamespace(
        id=sid, centroid=np.asarray(centroid, float), label=label, vertical=vertical
    )


def test_parse_levels_and_room():
    config = parse_config(config_data())
    assert config.frame == "map"
    assert config.names == ["cabinet", "table"]
    cabinet = config.get("cabinet")
    levels = cabinet.levels
    assert [lv.number for lv in levels] == [1, 2, 3, 4]
    assert levels[1].height == pytest.approx(0.42)
    assert levels[0].max_object_height == pytest.approx(0.42 - 0.02 - 0.05)
    assert levels[2].max_object_height == pytest.approx(1.12 - 0.02 - 0.78)
    assert levels[3].max_object_height is None
    assert cabinet.level(0) is None and cabinet.level(5) is None
    table = config.get("table")
    assert table.yaw == 0.0 and table.category == ""
    assert table.board_thickness == pytest.approx(0.02)


def test_footprint_rotation():
    cabinet = parse_config(config_data()).get("cabinet")
    # yaw 90 deg: local x (depth 0.35) along map y, width 0.80 along map x
    assert cabinet.contains([1.2 + 0.39, -0.4])
    assert not cabinet.contains([1.2, -0.4 + 0.2])
    assert cabinet.contains([1.2, -0.4 + 0.2], margin=0.05)
    corners = cabinet.corners()
    np.testing.assert_allclose(corners.min(0), [0.8, -0.575], atol=1e-9)
    np.testing.assert_allclose(corners.max(0), [1.6, -0.225], atol=1e-9)
    outline = cabinet.outline(2)
    assert outline.shape == (5, 3)
    np.testing.assert_allclose(outline[0], outline[-1])
    assert (outline[:, 2] == 0.42).all()
    np.testing.assert_allclose(cabinet.view_center(3), [1.2, -0.4, 0.78])
    np.testing.assert_allclose(
        cabinet.view_center(0)[2], np.mean([0.05, 0.42, 0.78, 1.12])
    )


@pytest.mark.parametrize(
    "change,text",
    [
        ({"levels": [0.4, 0.4]}, "strictly increasing"),
        ({"levels": [0.8, 0.4]}, "strictly increasing"),
        ({"levels": []}, "non-empty"),
        ({"size": {"x": 0.0, "y": 0.8}}, "size must be > 0"),
        ({"size": {"x": 0.3}}, "missing keys"),
        ({"pose": {"x": 1.0, "y": 0.0, "roll": 1.0}}, "unknown keys"),
        ({"levels": [0.4, 0.41], "board_thickness": 0.02}, "board_thickness"),
        ({"board_thickness": -0.01}, ">= 0"),
        ({"name": ""}, "non-empty string"),
        ({"levels": ["high"]}, "must be a number"),
        ({"colour": "red"}, "unknown keys"),
    ],
)
def test_invalid_furniture(change, text):
    with pytest.raises(FurnitureConfigError, match=text):
        parse_config(config_data(**change))


def test_invalid_config():
    data = config_data()
    data["furniture"][1]["name"] = "cabinet"
    with pytest.raises(FurnitureConfigError, match="duplicate"):
        parse_config(data)
    with pytest.raises(FurnitureConfigError, match="frame"):
        parse_config({**config_data(), "frame": ""})
    with pytest.raises(FurnitureConfigError, match="missing keys"):
        parse_config({"frame": "map"})
    with pytest.raises(FurnitureConfigError, match="mapping"):
        parse_config(None)


def test_load_yaml(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text(
        "frame: map\nfurniture:\n  - name: t\n    pose: {x: 0, y: 0}\n"
        "    size: {x: 1, y: 1}\n    levels: [0.7]\n"
    )
    assert load_config(path).get("t").heights == (0.7,)
    with pytest.raises(FurnitureConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    path.write_text("frame: [")
    with pytest.raises(FurnitureConfigError):
        load_config(path)


def test_arena_config_loads():
    config = load_config(ARENA / "rcjo2026_arena.yaml")
    assert config.frame == "world"
    shelf = config.get("shelf")
    assert shelf.heights == (0.02, 0.31, 0.61, 0.91, 1.20)
    assert shelf.level(3).max_object_height == pytest.approx(0.28)
    assert config.get("dining_table").heights == (0.80,)


def test_match_point_level_and_margin():
    config = parse_config(config_data())
    options = AssociationOptions(margin=0.10, height_tolerance=0.05)
    match = match_point(config, [1.2, -0.4, 0.80], options)
    assert (match.furniture, match.level) == ("cabinet", 3)
    assert match.height_error == pytest.approx(0.02)
    # Inside the margin (0.175 + 0.08 along map y), outside it, off height.
    assert match_point(config, [1.2, -0.4 + 0.255, 0.42], options).level == 2
    assert match_point(config, [1.2, -0.4 + 0.3, 0.42], options) is None
    assert match_point(config, [1.2, -0.4, 0.60], options) is None
    assert match_point(config, [-1.0, 0.3, 0.74], options).furniture == "table"


def test_match_point_closest_footprint_center_wins():
    data = config_data()
    data["furniture"][1].update(
        pose={"x": 1.2, "y": 0.0}, size={"x": 1.2, "y": 0.3}, levels=[0.78]
    )
    config = parse_config(data)
    # In both grown footprints (map y -0.25 .. -0.125) at a shared height:
    # the nearer center wins.
    assert match_point(config, [1.2, -0.22, 0.78]).furniture == "cabinet"
    assert match_point(config, [1.2, -0.15, 0.78]).furniture == "table"


def test_associate_surfaces_through_the_transform():
    config = parse_config(config_data())
    # odom = map shifted by (1, 0.5) and turned 90 deg: map = R odom + t
    c, s = 0.0, 1.0
    odom_to_map = np.eye(4)
    odom_to_map[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    odom_to_map[:3, 3] = [1.0, 0.5, 0.0]

    def in_odom(p):
        p = np.asarray(p, float)
        return odom_to_map[:3, :3].T @ (p - odom_to_map[:3, 3])

    surfaces = [
        surface(1, in_odom([1.2, -0.4, 0.42])),
        surface(2, in_odom([1.25, -0.35, 0.79])),
        surface(3, in_odom([1.2, -0.4, 0.0]), label="floor"),
        surface(4, in_odom([1.2, -0.4, 0.42]), label="wall", vertical=True),
        surface(5, in_odom([-1.0, 0.0, 0.75]), label="table"),
        surface(6, in_odom([3.0, 3.0, 0.75]), label="table"),
        surface(0, in_odom([1.2, -0.4, 1.12])),
    ]
    matches = associate_surfaces(config, surfaces, odom_to_map)
    assert {k: (m.furniture, m.level) for k, m in matches.items()} == {
        1: ("cabinet", 2),
        2: ("cabinet", 3),
        5: ("table", 1),
    }


def test_resolve_request_statuses():
    config = parse_config(config_data())
    assert resolve_request(None, "cabinet", 1, 0.1).status == PlaceStatus.ERROR
    unknown = resolve_request(config, "fridge", 0, 0.1)
    assert unknown.status == PlaceStatus.ERROR and "cabinet, table" in unknown.message
    assert resolve_request(config, "cabinet", 5, 0.1).status == PlaceStatus.ERROR
    assert resolve_request(config, "cabinet", -1, 0.1).status == PlaceStatus.ERROR
    ok = resolve_request(config, "cabinet", 2, 0.30)
    assert ok.status is None and [lv.number for lv in ok.levels] == [2]
    assert ok.text == "cabinet level 2"
    tall = resolve_request(config, "cabinet", 2, 0.40)
    assert tall.status == PlaceStatus.NO_SPACE
    assert "0.40 m" in tall.message and "0.34 m" in tall.message
    # Any level: those with room; the top level has no limit.
    any_level = resolve_request(config, "cabinet", 0, 0.345)
    assert [lv.number for lv in any_level.levels] == [1, 4]
    assert resolve_request(config, "table", 0, 2.0).levels[0].number == 1


def test_requested_surfaces_and_missing_message():
    config = parse_config(config_data())
    matches = {
        3: match_point(config, [1.2, -0.4, 0.42]),
        4: match_point(config, [1.2, -0.4, 0.78]),
        7: match_point(config, [1.0, -0.4, 0.42]),
        9: match_point(config, [-1.0, 0.0, 0.75]),
    }
    level2 = resolve_request(config, "cabinet", 2, 0.1)
    assert requested_surfaces(level2, matches, [0, 3, 4, 7, 9]) == [3, 7]
    assert requested_surfaces(level2, matches, [3, 4, 7], [7, 9]) == [7]
    assert requested_surfaces(level2, matches, [4, 9]) == []
    every = resolve_request(config, "cabinet", 0, 0.1)
    assert requested_surfaces(every, matches, [3, 4, 9]) == [3, 4]
    text = missing_message(level2, matches)
    assert "cabinet level 2 is not in the latest detection" in text
    assert "S3, S7" in text
    level4 = resolve_request(config, "cabinet", 4, 0.1)
    assert "no surface associated" in missing_message(level4, matches)
    assert "surface_ids [9]" in missing_message(level2, matches, [9])


def test_level_markers():
    config = parse_config(config_data())
    markers = level_markers(config)
    assert [(m.furniture, m.number) for m in markers] == [
        ("cabinet", 1),
        ("cabinet", 2),
        ("cabinet", 3),
        ("cabinet", 4),
        ("table", 1),
    ]
    np.testing.assert_allclose(markers[4].center, [-1.0, 0.0, 0.75])
    assert markers[1].outline.shape == (5, 3)


def test_clip_plane_cuts_the_bleeding_hull_to_the_footprint():
    # A board 0.4 x 0.6 m at 0.61 m whose hull bled 5 cm towards the camera
    # (x -) and 6 cm to the side.
    x, y = np.meshgrid(np.arange(0.45, 0.90, 0.01), np.arange(-0.55, 0.11, 0.01))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.61)])
    plane = make_plane(
        points, np.arange(len(points)), (0.0, 0.0, 1.4), np.array([0.0, 0, 1])
    )
    board = parse_config(
        {
            "frame": "base",
            "furniture": [
                {
                    "name": "shelf",
                    "pose": {"x": 0.70, "y": -0.25},
                    "size": {"x": 0.4, "y": 0.6},
                    "levels": [0.31, 0.61],
                }
            ],
        }
    ).get("shelf")
    clipped = clip_plane(plane, board.corners(0.02))
    hull = clipped.hull_points()
    assert hull[:, 0].min() == pytest.approx(0.48, abs=1e-6)
    assert hull[:, 1].max() == pytest.approx(0.07, abs=1e-6)
    np.testing.assert_allclose(hull[:, 2], 0.61, atol=1e-9)
    assert clipped.area < plane.area and len(clipped.inliers) == len(points)
    # Nothing left (a wrong association).
    far = board.corners(0.0) + [5.0, 0.0]
    assert clip_plane(plane, far) is None
