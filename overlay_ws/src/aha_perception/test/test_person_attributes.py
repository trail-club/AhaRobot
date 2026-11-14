"""Posture and gesture rules, keypoint lifting and the VLM's attribute answers."""

import json

import numpy as np
import pytest

from aha_perception.person_attributes import (
    INDEX,
    KEYPOINTS,
    Label,
    Sample,
    attributes_prompt,
    classify_gesture,
    classify_posture,
    decide,
    decide_gesture,
    keypoints_in_mask,
    lift_keypoints,
    motion_unknown,
    normalize_garment_type,
    parse_attributes,
    swap_side,
    wait_for_motion,
)

# A person at the origin facing +x, so their left is +y; heights above the floor.
STANDING = {
    "nose": (0.08, 0.0, 1.62),
    "left_eye": (0.06, 0.03, 1.65),
    "right_eye": (0.06, -0.03, 1.65),
    "left_ear": (0.0, 0.07, 1.63),
    "right_ear": (0.0, -0.07, 1.63),
    "left_shoulder": (0.0, 0.18, 1.42),
    "right_shoulder": (0.0, -0.18, 1.42),
    "left_elbow": (0.0, 0.2, 1.15),
    "right_elbow": (0.0, -0.2, 1.15),
    "left_wrist": (0.02, 0.2, 0.9),
    "right_wrist": (0.02, -0.2, 0.9),
    "left_hip": (0.0, 0.1, 0.9),
    "right_hip": (0.0, -0.1, 0.9),
    "left_knee": (0.02, 0.1, 0.48),
    "right_knee": (0.02, -0.1, 0.48),
    "left_ankle": (0.0, 0.1, 0.08),
    "right_ankle": (0.0, -0.1, 0.08),
}


def sample(points=None, stamp=0.0, hidden=(), camera=None, mirror=False, **moved):
    """A Sample of STANDING with some keypoints moved or hidden.

    mirror: the labels a mirrored pose estimate would give (left and right swapped).
    """
    layout = dict(points or STANDING, **moved)
    names = [swap_side(n) if mirror else n for n in KEYPOINTS]
    array = np.array([layout[name] for name in names], dtype=float)
    scores = np.array([0.0 if name in hidden else 0.9 for name in names])
    return Sample(stamp, array, scores, None if camera is None else np.array(camera))


FRONT, BEHIND = (3.0, 0.0, 1.4), (-3.0, 0.0, 1.4)  # cameras; the person faces +x
FACE = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def sitting():
    layout = {}
    for name, (x, y, z) in STANDING.items():
        if name.endswith(("knee", "ankle")):
            layout[name] = (0.45, y, 0.5 if name.endswith("knee") else 0.08)
        elif name.endswith("hip"):
            layout[name] = (x, y, 0.5)
        else:
            layout[name] = (x, y, z - 0.42)
    return layout


def lying():
    """STANDING turned onto its back on a 0.5 m bed, head towards -x."""
    return {name: (-z, y, 0.5 + x) for name, (x, y, z) in STANDING.items()}


@pytest.mark.parametrize(
    "points, head_z, expected",
    [
        (STANDING, 1.75, "standing"),
        (sitting(), 1.33, "sitting"),
        (lying(), 0.6, "lying"),
    ],
)
def test_posture_from_keypoints(points, head_z, expected):
    label = classify_posture(sample(points), head_z)
    assert label.value == expected and label.confident


def test_posture_with_hidden_knees_uses_the_hips():
    hidden = ("left_knee", "right_knee", "left_ankle", "right_ankle")
    label = classify_posture(sample(sitting(), hidden=hidden), 1.33)
    assert (label.value, label.confident) == ("sitting", True)
    label = classify_posture(sample(hidden=hidden), 1.75)
    assert (label.value, label.confident) == ("standing", True)


def test_posture_without_keypoints_uses_the_head_height():
    assert classify_posture(None, 1.7) == Label("standing", False, "head at 1.70 m")
    # Even sitting on the floor puts the head higher than 0.8 m.
    lying = classify_posture(None, 0.5)
    assert (lying.value, lying.confident) == ("lying", True)
    assert classify_posture(None, 1.3) is None
    assert classify_posture(None, None) is None


def raised(side):
    return {
        f"{side}_elbow": (0.0, 0.25 if side == "left" else -0.25, 1.65),
        f"{side}_wrist": (0.0, 0.25 if side == "left" else -0.25, 1.9),
    }


@pytest.mark.parametrize("side", ["left", "right"])
def test_raised_arm_held_still(side):
    history = [sample(stamp=t, **raised(side)) for t in (0.0, 0.5, 1.0, 1.5)]
    label = classify_gesture(history)
    assert (label.value, label.confident) == (f"raising_{side}_arm", True)
    assert not classify_gesture(history[:3]).confident
    # One image cannot tell a raised hand from a waving one.
    assert not classify_gesture(history[-1:]).confident


def test_both_arms_raised():
    label = classify_gesture([sample(**raised("left"), **raised("right"))])
    assert label.value == "raising_both_arms"


def test_waving_is_a_swinging_raised_hand():
    history = []
    for i, t in enumerate((0.0, 0.3, 0.6, 0.9)):
        dy = 0.12 if i % 2 else -0.12
        history.append(
            sample(
                stamp=t,
                right_elbow=(0.0, -0.4, 1.5),
                right_wrist=(0.0, -0.4 + dy, 1.75),
            )
        )
    label = classify_gesture(history)
    assert (label.value, label.confident) == ("waving", True)


@pytest.mark.parametrize(
    "moved, expected",
    [
        # The person's own left arm out to their left.
        ({"left_elbow": (0.0, 0.45, 1.42), "left_wrist": (0.0, 0.72, 1.45)}, "left"),
        (
            {"right_elbow": (0.0, -0.45, 1.42), "right_wrist": (0.05, -0.72, 1.4)},
            "right",
        ),
        # The right arm across the body still points to the left.
        ({"right_elbow": (0.2, 0.0, 1.42), "right_wrist": (0.25, 0.45, 1.42)}, "left"),
    ],
)
def test_pointing_side_is_the_persons_own(moved, expected):
    label = classify_gesture([sample(**moved)])
    assert (label.value, label.confident) == (f"pointing_{expected}", True)


def test_pointing_side_follows_the_body_not_the_frame():
    """The same pose turned to face -x: the person's left is now -y."""
    moved = {"left_elbow": (0.0, 0.45, 1.42), "left_wrist": (0.0, 0.72, 1.45)}
    turned = {name: (-x, -y, z) for name, (x, y, z) in dict(STANDING, **moved).items()}
    assert classify_gesture([sample(turned)]).value == "pointing_left"


def test_no_gesture_rules_for_a_lying_person():
    """Arms beside a lying body are level and far from the shoulders."""
    assert classify_gesture([sample(lying())]) is None


def test_one_image_with_swapped_sides_does_not_flip_the_gesture():
    left = {"left_elbow": (0.0, 0.45, 1.42), "left_wrist": (0.0, 0.72, 1.45)}
    swapped = {
        "left_shoulder": STANDING["right_shoulder"],
        "right_shoulder": STANDING["left_shoulder"],
        "right_elbow": left["left_elbow"],
        "right_wrist": left["left_wrist"],
        "left_elbow": STANDING["right_elbow"],
        "left_wrist": STANDING["right_wrist"],
    }
    history = [sample(stamp=0.0, **left), sample(stamp=0.5, **left)]
    history.append(sample(stamp=1.0, **swapped))
    label = classify_gesture(history)
    assert (label.value, label.confident) == ("pointing_left", True)
    assert "2/3" in label.reason


def test_mirrored_labels_are_swapped_back_for_a_person_facing_the_camera():
    history = [
        sample(stamp=s, camera=FRONT, mirror=True, **raised("left"))
        for s in (0.0, 0.5, 1.0, 1.5)
    ]
    assert classify_gesture(history).value == "raising_left_arm"
    # Without the camera position the mirrored labels are believed.
    blind = [sample(stamp=s, mirror=True, **raised("left")) for s in (0.0, 0.5)]
    assert classify_gesture(blind).value == "raising_right_arm"


def test_labels_are_kept_without_a_clearly_seen_face():
    """The face may be turned away or only cut off at the image edge."""
    for camera in (FRONT, BEHIND):
        correct = sample(camera=camera, hidden=FACE[:3], **raised("left"))
        assert classify_gesture([correct]).value == "raising_left_arm"


def test_both_arms_held_still_are_not_waving():
    both = {**raised("left"), **raised("right")}
    history = [sample(stamp=s, camera=FRONT, **both) for s in (0.0, 0.5, 1.0, 1.5)]
    label = classify_gesture(history)
    assert (label.value, label.confident) == ("raising_both_arms", True)


def test_waving_is_found_even_if_the_latest_image_misses_the_hand():
    history = []
    for i, t in enumerate((0.0, 0.3, 0.6)):
        dy = 0.12 if i % 2 else -0.12
        history.append(
            sample(
                stamp=t,
                camera=FRONT,
                right_elbow=(0.0, -0.4, 1.5),
                right_wrist=(0.0, -0.4 + dy, 1.75),
            )
        )
    history.append(sample(stamp=0.9, camera=FRONT, hidden=("right_wrist",)))
    assert classify_gesture(history).value == "waving"


def test_labels_swapped_between_images_are_not_a_waving_hand():
    history = [
        sample(stamp=s, camera=FRONT, mirror=i % 2 == 1, **raised("left"))
        for i, s in enumerate((0.0, 0.5, 1.0, 1.5))
    ]
    label = classify_gesture(history)
    assert (label.value, label.confident) == ("raising_left_arm", True)


def test_a_level_forearm_does_not_wave():
    """Depth noise moves a pointing hand sideways; it is not a raised hand."""
    history = []
    for i, t in enumerate((0.0, 0.5, 1.0, 1.5)):
        dy = 0.08 if i % 2 else -0.08
        history.append(
            sample(
                stamp=t,
                left_elbow=(0.0, 0.45, 1.42),
                left_wrist=(0.0, 0.72 + dy, 1.47),
            )
        )
    assert classify_gesture(history).value == "pointing_left"


def test_the_vlm_does_not_change_the_side_of_a_raised_arm():
    unsure = Label("raising_left_arm", False)
    assert decide_gesture(unsure, "raising_right_arm") == (
        "raising_left_arm",
        "keypoints",
    )
    assert decide_gesture(unsure, "waving") == ("waving", "vlm")
    assert decide_gesture(unsure, None) == ("raising_left_arm", "keypoints")
    assert decide_gesture(Label("none", False), "pointing_left") == (
        "pointing_left",
        "vlm",
    )
    assert decide_gesture(Label("waving", True), "none") == ("waving", "keypoints")
    assert decide_gesture(None, "waving") == ("waving", "vlm")


def test_no_gesture_and_hidden_arms():
    none = classify_gesture([sample()])
    assert (none.value, none.confident) == ("none", True)
    hidden = classify_gesture([sample(hidden=("left_wrist",))])
    assert (hidden.value, hidden.confident) == ("none", False)
    assert classify_gesture([sample(hidden=("left_wrist", "right_wrist"))]) is None
    assert classify_gesture([]) is None


def test_lift_keypoints_uses_depth_near_the_body():
    depth = np.full((40, 60), 3.0, np.float32)
    depth[10:30, 15:25] = 2.0
    mask = np.zeros((40, 60), bool)
    mask[10:30, 15:25] = True
    keypoints = np.zeros((17, 3))
    keypoints[:, 2] = 0.9
    keypoints[:] = (20.0, 20.0, 0.9)
    keypoints[INDEX["left_wrist"]] = (36.0, 20.0, 0.9)  # outside the mask
    keypoints[INDEX["nose"]] = (20.0, 12.0, 0.1)  # not seen
    points, scores = lift_keypoints(keypoints, mask, depth, 20.0, 20.0, 20.0, 20.0)
    np.testing.assert_allclose(points[INDEX["left_shoulder"]], (0.0, 0.0, 2.0))
    # Background depth is 1 m behind: the body's median depth is used instead.
    np.testing.assert_allclose(points[INDEX["left_wrist"]], (1.6, 0.0, 2.0))
    assert scores[INDEX["nose"]] == 0.0


def test_keypoints_at_the_image_edge_are_not_seen():
    depth = np.full((40, 60), 2.0, np.float32)
    mask = np.ones((40, 60), bool)
    keypoints = np.tile([30.0, 20.0, 0.9], (17, 1))
    keypoints[INDEX["left_wrist"]] = (57.0, 20.0, 0.9)  # cut off at the right edge
    _, scores = lift_keypoints(keypoints, mask, depth, 20.0, 20.0, 30.0, 20.0)
    assert scores[INDEX["left_wrist"]] == 0.0
    assert scores[INDEX["nose"]] > 0.0


def test_keypoints_in_mask_picks_the_part_inside():
    mask = np.zeros((20, 40), bool)
    mask[:, 20:] = True
    left = np.tile([5.0, 10.0, 0.9], (17, 1))
    right = np.tile([30.0, 10.0, 0.9], (17, 1))
    np.testing.assert_array_equal(keypoints_in_mask([left, right], mask), right)
    assert keypoints_in_mask([left], mask) is None


def test_parse_attributes_keeps_known_values():
    answer = parse_attributes(
        "Sure: "
        + json.dumps(
            {
                "posture": "Sitting",
                "gesture": "raising left arm",
                "upper_garments": [
                    {"type": "T-shirt", "colour": "Grey"},
                    {"type": "cardigan", "color": "red"},
                    {"type": "jacket", "colour": "teal"},
                    {"type": "t shirt", "colour": "gray"},
                ],
            }
        )
    )
    assert answer.posture == "sitting"
    assert answer.gesture == "raising_left_arm"
    assert answer.garments == ["gray t shirt", "red top"]
    bad = parse_attributes('{"posture": "dancing", "gesture": 3}')
    assert (bad.posture, bad.gesture, bad.garments) == (None, None, [])


def test_garment_spellings():
    assert normalize_garment_type("tshirts") == "t shirt"
    assert normalize_garment_type("Jackets") == "jacket"
    assert normalize_garment_type("hoodie") is None


def test_prompt_mentions_motion_only_for_several_photos():
    assert "cannot show" in attributes_prompt(1, 0.0)
    assert "Photos 1 to 3" in attributes_prompt(3, 0.5)


def test_decide_prefers_confident_rules_then_the_vlm():
    assert decide(Label("sitting", True), "standing", "geometry") == (
        "sitting",
        "geometry",
    )
    assert decide(Label("sitting", False), "standing", "geometry") == (
        "standing",
        "vlm",
    )
    assert decide(Label("sitting", False), None, "geometry") == ("sitting", "geometry")
    assert decide(None, None, "geometry") == ("", "")


def test_only_an_unsure_raised_hand_waits_for_motion():
    assert motion_unknown(Label("raising_left_arm", False))
    assert not motion_unknown(Label("raising_left_arm", True))
    assert not motion_unknown(Label("pointing_left", False))
    assert not motion_unknown(Label("none", False))
    assert not motion_unknown(None)


def test_classify_waits_for_motion_only_with_the_vlm():
    unsure = Label("raising_left_arm", False)
    assert wait_for_motion(False, unsure, 1.0, 3.0)
    assert not wait_for_motion(True, unsure, 0.0, 3.0)  # skip_vlm answers at once
    assert not wait_for_motion(False, unsure, 3.0, 3.0)  # waited long enough
    assert not wait_for_motion(False, Label("waving", True), 0.0, 3.0)
