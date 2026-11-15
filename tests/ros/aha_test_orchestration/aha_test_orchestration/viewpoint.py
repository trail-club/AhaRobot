"""Where to stand and look to observe a whole furniture (ROS free).

From close up the board above hides the back of a lower level: the ray from
the camera (height H) to the back edge of a level (top h, depth D behind the
front face) passes under the next board's bottom b at the front face only if
the camera is at least ``(H - b) * D / (b - h)`` from the front face
(``required_distance``). ``observe_pose`` moves the base straight back from
the furniture face it is in front of, keeping its lateral offset and
heading, to the largest such distance over the observed levels (plus a
margin, at most ``max_distance``). ``view_points`` then groups the levels into
as few head directions as fit the camera's vertical field of view (each
level from its front edge to its back edge, up to ``object_height`` above
it) and returns a look point per group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ViewOptions:
    margin: float = 0.05  # m added to the required camera distance
    min_distance: float = 0.0  # m, camera to the front face
    max_distance: float = 1.5
    # 640x480 color of the head camera: about 43.4 deg (sensors.xacro, the
    # object detection uses the color image); minus a margin per side.
    vertical_fov: float = math.radians(43.4)
    fov_margin: float = math.radians(4.0)
    object_height: float = 0.2  # m above a level kept in view (its objects)


def required_distance(heights, board_thickness, depth, camera_height, levels):
    """Camera distance to the front face (m) from which the back edges of
    ``levels`` (numbers, 1 = lowest) are seen under the boards above them."""
    need = 0.0
    for number in levels:
        k = number - 1
        if k + 1 >= len(heights) or heights[k] >= camera_height:
            continue  # top level, or a level the camera cannot see from above
        top = heights[k]
        bottom = heights[k + 1] - board_thickness
        if camera_height <= bottom or bottom <= top:
            continue
        need = max(need, (camera_height - bottom) * depth / (bottom - top))
    return need


def front_face(furniture, xy):
    """(outward unit normal (2,), half extent along it) of the footprint face
    the point ``xy`` (config frame) is in front of."""
    local = furniture.to_local(xy)
    half = np.array([furniture.depth / 2, furniture.width / 2])
    axis = int(np.argmax(np.abs(local) / half))
    sign = 1.0 if local[axis] >= 0 else -1.0
    c, s = math.cos(furniture.yaw), math.sin(furniture.yaw)
    axes = np.array([[c, s], [-s, c]])  # local x and y in the config frame
    return sign * axes[axis], float(half[axis])


def observe_pose(furniture, pose, camera_forward, camera_height, levels, options):
    """(base pose (x, y, yaw, config frame), camera distance to the face,
    furniture depth behind the face): straight back from the face in front
    of ``pose`` (x, y, yaw) to the distance from which ``levels`` are seen
    whole. ``camera_forward`` is the camera's x in the base frame (m), along
    the base heading."""
    normal, half = front_face(furniture, pose[:2])
    need = required_distance(
        furniture.heights,
        furniture.board_thickness,
        2 * half,
        camera_height,
        levels,
    )
    camera = min(max(need + options.margin, options.min_distance), options.max_distance)
    # Base distance to the face along the normal; the camera is ahead of the
    # base along the heading, which faces the furniture.
    facing = max(-(math.cos(pose[2]) * normal[0] + math.sin(pose[2]) * normal[1]), 0.0)
    base = camera + camera_forward * facing
    offset = np.asarray(pose[:2], dtype=float) - furniture.center
    lateral = offset - (offset @ normal) * normal
    xy = furniture.center + lateral + normal * (half + base)
    return (float(xy[0]), float(xy[1]), float(pose[2])), camera, 2 * half


def level_angles(top, depth, camera_distance, camera_height, object_height):
    """(min, max) depression angles (rad, down positive) of a level from its
    front edge to its back edge, up to ``object_height`` above it."""
    angles = [
        math.atan2(camera_height - z, x)
        for x in (camera_distance, camera_distance + depth)
        for z in (top, top + object_height)
    ]
    return min(angles), max(angles)


def view_points(furniture, camera_distance, depth, camera_height, levels, options):
    """[(look point (3,) config frame, level numbers)] covering ``levels``
    from the camera distance and furniture depth of ``observe_pose``: levels
    grouped from the top while their depression angles fit the field of
    view; each point is on the furniture's vertical center line at the
    group's middle angle."""
    span = options.vertical_fov - 2 * options.fov_margin
    groups = []  # [low, high, numbers]
    for number in sorted(levels, reverse=True):
        low, high = level_angles(
            furniture.heights[number - 1],
            depth,
            camera_distance,
            camera_height,
            options.object_height,
        )
        if groups and max(high, groups[-1][1]) - min(low, groups[-1][0]) <= span:
            last = groups[-1]
            groups[-1] = [min(low, last[0]), max(high, last[1]), last[2] + [number]]
        else:
            groups.append([low, high, [number]])
    center = camera_distance + depth / 2
    return [
        (
            np.array(
                [
                    furniture.x,
                    furniture.y,
                    camera_height - math.tan((low + high) / 2) * center,
                ]
            ),
            tuple(numbers),
        )
        for low, high, numbers in groups
    ]
