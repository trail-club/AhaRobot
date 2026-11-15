import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from aha_sim_tasks import laundry_pattern
from aha_sim_tasks.api import Action
from aha_sim_tasks.evaluation import load_tasks, make_evaluator
from aha_sim_tasks.laundry_evaluation import (
    LaundryEvaluator,
    LaundryState,
    fold_quality,
    parse_state,
)
from aha_sim_tasks.laundry_scene import build_world, sample_laundry_scene
from aha_sim_tasks.policy_protocol import action_message, decode_action

CONFIG = Path(__file__).resolve().parents[1] / "config"
FOLD = {
    "flat_height": 0.04,
    "bunched_height": 0.06,
    "max_side_ratio": 0.75,
    "min_folded_fraction": 0.4,
    "max_speed": 0.05,
}


def base_scene():
    return json.loads((CONFIG / "laundry_scene.json").read_text())


def laundry_task():
    scene = sample_laundry_scene(base_scene(), seed=0)
    return load_tasks(CONFIG / "tasks.json", scene)["doing_laundry"]


def signed_area(points):
    return (
        sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
        / 2
    )


def test_pattern_panels_share_their_seams():
    front, front_seams = laundry_pattern.outline(
        laundry_pattern.BASE["front_neck_drop"]
    )
    back, back_seams = laundry_pattern.outline(laundry_pattern.BASE["back_neck_drop"])
    assert signed_area(front) > 0 and signed_area(back) > 0
    sewn = lambda points, seams: {  # noqa: E731
        (tuple(points[k]), tuple(points[(k + 1) % len(points)]))
        for k, seam in enumerate(seams)
        if seam
    }
    assert sewn(front, front_seams) == sewn(back, back_seams)
    width, length = laundry_pattern.extent(laundry_pattern.SIZES["L"])
    assert length == pytest.approx(0.72 * 1.08)
    assert laundry_pattern.pattern("XL")["mass"] > laundry_pattern.pattern("S")["mass"]


def test_scene_sampling_follows_the_rulebook_counts_and_seed():
    scene = sample_laundry_scene(base_scene(), seed=4)
    origins = [item["origin"] for item in scene["laundry_items"]]
    assert origins.count("washer") == 3 and origins.count("basket") == 7
    assert origins.count("table") == 1
    assert scene == sample_laundry_scene(base_scene(), seed=4)
    for seed in range(20):
        varied = sample_laundry_scene(base_scene(), seed=seed, randomize=True)
        origins = [item["origin"] for item in varied["laundry_items"]]
        assert 2 <= origins.count("washer") <= 4 and 6 <= origins.count("basket") <= 8
    garment = scene["laundry"]["garments"][0]
    assert Path(garment["pattern"]).is_file() and Path(garment["albedo"]).is_file()
    assert len(garment["color"]) == 3 and all(0 <= c <= 1 for c in garment["color"])
    assert scene["laundry"]["regions"]["washer"]["door_model"] == "washing_machine_door"


def test_world_contains_the_laundry_room(tmp_path):
    scene = sample_laundry_scene(base_scene(), seed=1, randomize=True)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    root = ET.fromstring(build_world(scene, path, cameras=False))
    models = {model.get("name") for model in root.iter("model")}
    assert {
        "washing_machine",
        "washing_machine_door",
        "laundry_basket",
        "folding_table",
        "arena",
        "arena_door",
        "venue",
        "ground_plane",
    } <= models
    plugin = next(p for p in root.iter("plugin") if p.get("filename") == "aha_laundry")
    assert plugin.findtext("scene_config") == str(path)
    assert (tmp_path / "arena_floor.obj").is_file()
    assert not any(
        p.get("filename") == "gz-sim-sensors-system" for p in root.iter("plugin")
    )


def fold(**overrides):
    neat = {
        "area_ratio": 0.3,
        "length_ratio": 0.6,
        "width_ratio": 0.4,
        "rectangularity": 0.94,
        "height_p95": 0.05,
        "expected_height": 0.02,
        "layer_cv": 0.15,
        "folded_fraction": 0.9,
        "mean_layers": 6.0,
    }
    return {**neat, **overrides}


@pytest.mark.parametrize(
    "metrics, expected",
    [
        (fold(), 1.0),
        # One fold in half with the sleeves out: acceptable but not neat.
        (
            fold(
                area_ratio=0.56,
                length_ratio=1.0,
                width_ratio=0.49,
                rectangularity=0.76,
                height_p95=0.043,
                expected_height=0.012,
                layer_cv=0.27,
                folded_fraction=0.73,
            ),
            0.5,
        ),
        # A flat, spread shirt was never folded.
        (
            fold(
                area_ratio=1.0, length_ratio=1.0, width_ratio=1.0, folded_fraction=0.0
            ),
            0.0,
        ),
        # A crumpled pile is small but bunched.
        (
            fold(area_ratio=0.3, height_p95=0.19, expected_height=0.017, layer_cv=0.5),
            0.0,
        ),
    ],
)
def test_fold_quality_maps_geometry_to_appendix_a_levels(metrics, expected):
    fraction, _ = fold_quality(metrics, FOLD)
    assert fraction == expected
    assert fold_quality(None, FOLD)[0] == 0.0


def garment(gid, origin, region, **extra):
    fractions = {"washer": 0.0, "basket": 0.0, "table": 0.0, "floor": 0.0}
    if region:
        fractions[region] = 1.0
    return {
        "id": gid,
        "origin": origin,
        "fractions": fractions,
        "held_by": {},
        "touched_floor": False,
        "speed": 0.0,
        "fold": None,
        **extra,
    }


def state(
    t, garments, robot=(-0.9, 0.0, 0.0), door=0.0, basket=(3.78, -0.98), stacks=()
):
    return LaundryState(
        sim_time=t,
        robot_pose=robot,
        started=True,
        washer_door_angle=door,
        basket_pose=(*basket, 0.0, 0.0, 0.0, 0.0, 1.0),
        garments=tuple(garments),
        stacks=tuple(stacks),
    )


def test_score_sheet_latches_events_and_rules_on_final_folds():
    task = laundry_task()
    evaluator = make_evaluator(task, 10.0)
    assert isinstance(evaluator, LaundryEvaluator)
    start = [
        garment("w0", "washer", "washer"),
        garment("b0", "basket", "basket"),
        garment("b1", "basket", "basket"),
    ]
    assert evaluator.update(state(10.0, start)) is None
    assert evaluator.metrics["score"] == 0
    # Into the laundry area, door open, one basket shirt lifted out.
    picked = [
        garment("w0", "washer", "washer"),
        garment("b0", "basket", None, held_by={"right": 3}),
        garment("b1", "basket", "basket"),
    ]
    evaluator.update(state(20.0, picked, robot=(3.0, 0.0, 0.0), door=1.2))
    assert evaluator.metrics["items"] == {
        "navigation": 15,
        "washer_door": 300,
        "basket_pick": 100,
    }
    # The washer shirt comes out but touches the floor.
    dropped = [
        garment("w0", "washer", "floor", touched_floor=True),
        garment("b0", "basket", "table"),
        garment("b1", "basket", "basket"),
    ]
    evaluator.update(state(30.0, dropped, robot=(3.0, 0.0, 0.0)))
    assert evaluator.metrics["penalties"] == {"washer_floor": -200}
    # Two neatly folded shirts stacked on the table end the episode.
    neat = fold()
    final = [
        garment("w0", "washer", "table", fold=neat),
        garment("b0", "basket", "table", fold=neat),
        garment("b1", "basket", "basket"),
    ]
    stacks = [{"below": "b0", "above": "w0", "overlap": 0.9}]
    evaluator.update(state(40.0, final, robot=(3.0, 0.0, 0.0), stacks=stacks))
    assert evaluator.finish(ended=True) == "success"
    items = evaluator.metrics["items"]
    assert items["folding"] == 800 and items["additional_folding"] == 400
    assert items["stacking"] == 200 and items["washer_removal"] == 300
    assert evaluator.metrics["score"] == 15 + 300 + 100 + 300 + 800 + 400 + 200 - 200


def test_multiple_pick_transport_and_time_limit():
    task = laundry_task()
    evaluator = LaundryEvaluator(task, 0.0)
    start = [
        garment("b0", "basket", "basket"),
        garment("b1", "basket", "basket"),
        garment("w0", "washer", "washer"),
        garment("t0", "table", "floor"),  # never in its place: cannot score
    ]
    evaluator.update(state(0.5, start))
    both = [
        garment("b0", "basket", None, held_by={"right": 2}),
        garment("b1", "basket", None, held_by={"right": 1}),
        garment("w0", "washer", "basket"),
        garment("t0", "table", "floor"),
    ]
    evaluator.update(state(1.0, both))
    assert evaluator.metrics["penalties"] == {"multiple_pick": -100}
    assert evaluator.metrics["items"] == {"basket_pick": 100, "washer_removal": 300}
    carried = [
        garment("b0", "basket", "table"),
        garment("b1", "basket", "table"),
        garment("w0", "washer", "basket"),
        garment("t0", "table", "floor"),
    ]
    # The basket travels next to the table with the washer shirt inside.
    evaluator.update(state(2.0, carried, basket=(3.4, 0.4)))
    assert "basket_transport" in evaluator.metrics["items"]
    assert (
        evaluator.update(state(task.timeout + 1.0, carried, basket=(3.4, 0.4)))
        == "timeout"
    )
    assert evaluator.metrics["main_goal"] is False


def test_laundry_state_parsing_and_task_definition():
    task = laundry_task()
    assert task.criterion == "laundry" and task.timeout == 420.0
    points = task.parameters["points"]
    maximum = (
        points["navigation"]
        + points["basket_pick"]
        + points["folding"]
        + points["washer_door"]
        + points["washer_removal"]
        + points["basket_transport"]
        + 5 * points["additional_folding"]
        + 6 * points["stacking"]
    )
    assert maximum == task.parameters["max_score"] == 4415
    sample = {
        "schema_version": 3,
        "task": "laundry",
        "frame_id": "world",
        "stamp": {"sec": 3, "nanosec": 500000000},
        "robot_pose": None,
        "started": False,
        "washer_door_angle": 0.0,
        "basket_pose": [0, 0, 0, 0, 0, 0, 1],
        "garments": [garment("w0", "washer", "washer")],
        "stacks": [],
    }
    parsed = parse_state(sample)
    assert parsed.sim_time == 3.5 and parsed.robot_pose is None
    with pytest.raises(ValueError):
        parse_state({**sample, "garments": [{"id": "x"}]})


def test_actions_can_end_the_episode():
    action = Action(end_episode=True)
    action.validate()
    assert decode_action(action_message(action)).end_episode
    assert "end_episode" not in action_message(Action())
    with pytest.raises(ValueError):
        Action(end_episode=1).validate()
