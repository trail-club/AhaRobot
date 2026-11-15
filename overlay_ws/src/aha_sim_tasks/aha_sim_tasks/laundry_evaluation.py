"""Doing Laundry score sheet (RoboCup@Home rulebook 2027, 5.4 and Appendix A).

Scores come only from simulator state published by the laundry system:
garment regions, gripper attachments and the footprint geometry of garments
on the table. Human-assistance items do not occur in simulation.
"""

from dataclasses import asdict, dataclass, field
import json
import math

# Appendix A rulings for a robot without help, best first.
QUALITY_LEVELS = (
    (0.85, 1.0, "neat"),
    (0.72, 0.875, "unintended folds"),
    (0.60, 0.75, "good"),
    (0.42, 0.5, "basic"),
    (0.0, 0.375, "misaligned"),
)


@dataclass(frozen=True)
class LaundryState:
    sim_time: float
    robot_pose: tuple | None
    started: bool
    washer_door_angle: float | None
    basket_pose: tuple
    garments: tuple
    stacks: tuple
    grippers: dict = field(default_factory=dict)
    arena_door_angle: float | None = None
    robot_speed: float = 0.0
    object_speed: float = 0.0  # interface shared with object tasks


def parse_state(sample):
    """LaundryState from a schema 3 sample, validating what scoring uses."""
    if sample.get("task") != "laundry" or sample.get("frame_id") != "world":
        raise ValueError("Unsupported laundry evaluation state")
    stamp = sample["stamp"]
    sim_time = stamp["sec"] + stamp["nanosec"] * 1e-9
    robot = sample.get("robot_pose")
    basket = tuple(sample["basket_pose"])
    garments = tuple(sample["garments"])
    if (
        not math.isfinite(sim_time)
        or (
            robot is not None
            and (len(robot) != 3 or not all(map(math.isfinite, robot)))
        )
        or len(basket) != 7
        or not all(map(math.isfinite, basket))
    ):
        raise ValueError("Invalid laundry evaluation state")
    for garment in garments:
        if (
            not {"id", "origin", "fractions", "held_by", "touched_floor"}
            <= garment.keys()
        ):
            raise ValueError("Incomplete garment state")
    return LaundryState(
        sim_time=sim_time,
        robot_pose=tuple(robot) if robot is not None else None,
        started=bool(sample.get("started", False)),
        washer_door_angle=sample.get("washer_door_angle"),
        arena_door_angle=sample.get("arena_door_angle"),
        basket_pose=basket,
        garments=garments,
        stacks=tuple(sample.get("stacks", ())),
        grippers=sample.get("grippers", {}),
    )


def _clamp(value):
    return max(0.0, min(1.0, value))


def fold_quality(fold, settings):
    """Folding score fraction and its components from footprint geometry.

    A fold must reduce the shirt in at least one dimension and stack layers
    over much of the footprint; crumpled cloth is excluded by its height and
    uneven layers. Neatness combines flatness, rectangularity, layer
    uniformity and the number of folds, then maps to the Appendix A levels.
    """
    if not fold:
        return 0.0, {"folded": False, "reason": "not lying on the table"}
    excess = fold["height_p95"] - fold["expected_height"]
    components = {
        # The cloth mesh cannot make creases flatter than a few centimetres.
        "flatness": _clamp(
            1 - (excess - settings["flat_height"]) / settings["bunched_height"]
        ),
        "rectangularity": _clamp((fold["rectangularity"] - 0.70) / 0.22),
        "uniformity": _clamp(1 - (fold["layer_cv"] - 0.15) / 0.35),
        "compactness": _clamp((0.75 - fold["area_ratio"]) / 0.4),
    }
    reduced = (
        min(fold["length_ratio"], fold["width_ratio"]) <= settings["max_side_ratio"]
    )
    layered = fold["folded_fraction"] >= settings["min_folded_fraction"]
    if not (reduced and layered):
        return 0.0, {
            **components,
            "folded": False,
            "reason": "no fold reduces the shirt",
        }
    if components["flatness"] < 0.3 or fold["layer_cv"] > 0.45:
        return 0.0, {
            **components,
            "folded": False,
            "reason": "bunched rather than folded",
        }
    neatness = components["flatness"] * (
        0.4 * components["rectangularity"]
        + 0.3 * components["uniformity"]
        + 0.3 * components["compactness"]
    )
    for threshold, fraction, label in QUALITY_LEVELS:
        if neatness >= threshold:
            return fraction, {
                **components,
                "folded": True,
                "neatness": neatness,
                "ruling": label,
            }
    return 0.0, components  # unreachable: the last level starts at zero


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def distance_to_rectangle(point, center, yaw, half):
    dx, dy = point[0] - center[0], point[1] - center[1]
    c, s = math.cos(yaw), math.sin(yaw)
    u, v = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(abs(u) - half[0], 0.0), max(abs(v) - half[1], 0.0))


class LaundryEvaluator:
    """Latches score-sheet events and rules on folding at the episode end."""

    def __init__(self, task, start_time, trace=None):
        self.task = task
        self.start_time = start_time
        self.trace = trace
        self.p = task.parameters
        self.points = self.p["points"]
        self.events = {}
        self.penalties = {}
        self.result = None
        self.metrics = {}
        self.previous_time = None
        self.last_state = None
        self.basket_start = None
        self.transport_load = None
        self.loaded = set()  # garments seen in their starting container

    # ---------------------------------------------------------------- events
    def _latch(self, name, state, **details):
        if name not in self.events:
            self.events[name] = {"time": state.sim_time - self.start_time, **details}

    def _penalize(self, name, state, **details):
        if name not in self.penalties:
            self.penalties[name] = {"time": state.sim_time - self.start_time, **details}

    def _observe(self, state):
        if state.robot_pose is not None and point_in_polygon(
            state.robot_pose[:2], self.p["laundry_area"]
        ):
            self._latch("navigation", state)
        if (state.washer_door_angle or 0.0) >= self.p["door_open_angle"]:
            self._latch("washer_door", state, angle=state.washer_door_angle)
        basket_xy = state.basket_pose[:2]
        if self.basket_start is None:
            self.basket_start = basket_xy
        moved = math.dist(basket_xy, self.basket_start)
        held = [g for g in state.garments if g["held_by"]]
        for garment in state.garments:
            f = garment["fractions"]
            if f[garment["origin"]] >= 0.5:
                self.loaded.add(garment["id"])
            # Only laundry that was in its container can be taken out of it.
            if garment["id"] not in self.loaded:
                continue
            if (
                garment["origin"] == "basket"
                and garment["held_by"]
                and f["basket"] <= 0.05
            ):
                lifted = [
                    g
                    for g in held
                    if g["origin"] == "basket" and g["fractions"]["basket"] < 0.5
                ]
                self._latch("basket_pick", state, garment=garment["id"])
                if len(lifted) >= 2:
                    self._penalize(
                        "multiple_pick", state, garments=[g["id"] for g in lifted]
                    )
            if garment["origin"] == "washer" and f["washer"] == 0.0:
                self._latch("washer_removal", state, garment=garment["id"])
            if garment["origin"] == "washer" and garment["touched_floor"]:
                self._penalize("washer_floor", state, garment=garment["id"])
        # Basket transport: washer laundry carried in the basket to the table.
        loaded = [
            g["id"]
            for g in state.garments
            if g["origin"] == "washer"
            and g["fractions"]["basket"] >= self.p["basket_load_fraction"]
        ]
        if self.transport_load is None and moved >= 0.1:
            self.transport_load = {
                g["id"] for g in state.garments if g["fractions"]["basket"] >= 0.6
            }
        if self.transport_load and "basket_transport" not in self.events:
            for garment in state.garments:
                if garment["id"] in self.transport_load and garment["touched_floor"]:
                    self._penalize("transport_loss", state, garment=garment["id"])
        reach = distance_to_rectangle(
            basket_xy, self.p["table_center"], self.p["table_yaw"], self.p["table_half"]
        )
        if (
            loaded
            and moved >= self.p["basket_min_travel"]
            and reach <= self.p["basket_reach"]
        ):
            self._latch(
                "basket_transport", state, garments=loaded, distance_to_table=reach
            )

    # --------------------------------------------------------------- ruling
    def _rule_folding(self, state):
        settings = self.p["fold"]
        folds = {}
        for garment in state.garments:
            fraction, details = (0.0, {"folded": False, "reason": "held or moving"})
            if (
                not garment["held_by"]
                and garment.get("speed", 0.0) <= settings["max_speed"]
            ):
                fraction, details = fold_quality(garment.get("fold"), settings)
            folds[garment["id"]] = {"fraction": fraction, **details}
        ranked = sorted(
            (
                (value["fraction"], name)
                for name, value in folds.items()
                if value["fraction"] > 0
            ),
            reverse=True,
        )
        items = {}
        if ranked:
            items["folding"] = self.points["folding"] * ranked[0][0]
            extra = ranked[1 : 1 + self.p["max_additional_folds"]]
            items["additional_folding"] = sum(
                self.points["additional_folding"] * f for f, _ in extra
            )
        folded = {name for _, name in ranked}
        stacked = set()
        for stack in state.stacks:
            if (
                stack["overlap"] >= self.p["stack_overlap"]
                and stack["below"] in folded
                and stack["above"] in folded
            ):
                stacked |= {stack["below"], stack["above"]}
        if stacked:
            items["stacking"] = self.points["stacking"] * min(
                len(stacked), self.p["max_stacked"]
            )
        return items, folds, sorted(stacked)

    def _score(self, state):
        items = {name: self.points[name] for name in self.events if name in self.points}
        fold_items, folds, stacked = self._rule_folding(state)
        items.update(fold_items)
        penalties = {name: self.points[name] for name in self.penalties}
        total = sum(items.values()) + sum(penalties.values())
        return total, items, penalties, folds, stacked

    def _update_metrics(self, state):
        total, items, penalties, folds, stacked = self._score(state)
        self.metrics = {
            "elapsed_sim_seconds": state.sim_time - self.start_time,
            "score": total,
            "max_score": self.p["max_score"],
            "items": items,
            "penalties": penalties,
            "events": self.events,
            "penalty_events": self.penalties,
            "folds": folds,
            "stacked": stacked,
            "main_goal": "folding" in items,
            "robot_pose": state.robot_pose,
            "washer_door_angle": state.washer_door_angle,
            "basket_position": list(state.basket_pose[:3]),
            "garment_regions": {
                g["id"]: max(g["fractions"], key=g["fractions"].get)
                if max(g["fractions"].values()) > 0.5
                else "elsewhere"
                for g in state.garments
            },
        }

    def update(self, state):
        if self.result is not None:
            return self.result
        if not isinstance(state, LaundryState):
            raise TypeError("The laundry task needs laundry evaluation states")
        if self.previous_time is not None and state.sim_time <= self.previous_time:
            return None
        self.previous_time = state.sim_time
        self.last_state = state
        self._observe(state)
        elapsed = state.sim_time - self.start_time
        if self.trace is not None:
            self.trace.write(
                json.dumps(
                    {
                        "state": {**asdict(state), "garments": None},
                        "events": self.events,
                        "penalties": self.penalties,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        if elapsed >= self.task.timeout:
            return self.finish()
        # Folding cannot change once everything is achieved and at rest.
        self._update_metrics(state)
        if self.metrics["score"] >= self.p["max_score"]:
            return self.finish()
        return None

    def finish(self, state=None, ended=False):
        """Final ruling, at the time limit or when the team ends the run."""
        if self.result is not None:
            return self.result
        state = state or self.last_state
        if state is None:
            self.result = "error"
            return self.result
        self._update_metrics(state)
        self.metrics["ended_by_policy"] = ended
        if self.metrics["main_goal"]:
            self.result = "success"
        else:
            self.result = "ended" if ended else "timeout"
        return self.result
