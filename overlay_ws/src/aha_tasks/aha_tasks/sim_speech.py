"""Simulated voices answering the robot, without ROS: who speaks and what they say.

The simulation has no microphone, so `sim_speech` serves the ASR's Listen
action instead. The speaker is whoever the robot is close to: the operator
near the instruction point, or a named person of the people scene
(config/sim/people/) in front of the robot. The answer depends on the robot's
last sentence (its caption on the screen):

- asked for the first / second / third command: the operator says it; the
  next request of the same command in the same task run gets the next
  variant of the list (the referee rephrases, simpler each time);
- "... yes or no", "is that correct", "did you say": the operator says
  `confirm` (default "yes");
- "what is your name" / "who are you": a person says "My name is <name>.";
- "are you <name>": a person says "Yes." or "No, my name is <name>.".

Anything else gets no answer, and the Listen goal times out as with silence.

Config (config/sim/speech/<name>.json):
{"operator": {"location": "<task location>", "name": "<optional>",
 "confirm": "yes", "commands": ["<text>" or ["<variant>", ...], ...]},
 "radius": 1.5, "fov": 1.0, "delay": 1.0}
`radius` (m) is the distance at which someone answers, `fov` (rad) the half
angle in front of the robot within which a person answers (the operator
answers from any side), `delay` (s) the time before an answer.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

CONFIG_KEYS = {"operator", "radius", "fov", "delay"}
OPERATOR_KEYS = {"location", "commands", "name", "confirm"}
ORDINALS = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "second": 1,
    "2nd": 1,
    "two": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
}


@dataclass(frozen=True)
class Speaker:
    name: str  # "" for an operator without a name
    position: tuple  # (x, y) in map
    operator: bool


def load_config(path, locations, people=None):
    """(config, speakers) of a speech config, the task's locations and a people scene.

    `people` is the people scene dict (aha_tasks.sim_people.load_scene); its
    people with a "name" answer name questions.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) - CONFIG_KEYS or "operator" not in config:
        raise ValueError(f"{path}: needs operator, may have {sorted(CONFIG_KEYS)}")
    operator = config["operator"]
    if not isinstance(operator, dict) or set(operator) - OPERATOR_KEYS:
        raise ValueError(f"{path}: operator may have {sorted(OPERATOR_KEYS)}")
    location = operator.get("location")
    if location not in locations:
        raise ValueError(
            f"{path}: operator location {location!r} is not a task location"
        )
    commands = operator.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError(f"{path}: commands must be a list")
    variants = []
    for command in commands:
        options = [command] if isinstance(command, str) else command
        if (
            not isinstance(options, list)
            or not options
            or not all(isinstance(o, str) and o.strip() for o in options)
        ):
            raise ValueError(f"{path}: a command is a text or a list of texts")
        variants.append(options)
    settings = {
        "commands": variants,
        "confirm": str(operator.get("confirm", "yes")),
        "radius": float(config.get("radius", 1.5)),
        "fov": float(config.get("fov", 1.0)),
        "delay": float(config.get("delay", 1.0)),
    }
    x, y, _ = locations[location]
    speakers = [Speaker(str(operator.get("name", "")), (float(x), float(y)), True)]
    for person in (people or {}).get("people", {}).values():
        if person.get("name"):
            px, py, _ = person["position"]
            speakers.append(Speaker(person["name"], (float(px), float(py)), False))
    return settings, speakers


def speaker_for(pose, speakers, radius, fov):
    """The nearest speaker within `radius` of the robot pose (x, y, yaw), or None.

    People must be within `fov` of the robot's heading; the operator not.
    """
    x, y, yaw = pose
    best = None
    for speaker in speakers:
        dx, dy = speaker.position[0] - x, speaker.position[1] - y
        distance = math.hypot(dx, dy)
        if distance > radius:
            continue
        if not speaker.operator and distance > 1e-6:
            bearing = math.atan2(dy, dx) - yaw
            if abs(math.atan2(math.sin(bearing), math.cos(bearing))) > fov:
                continue
        if best is None or distance < best[0]:
            best = (distance, speaker)
    return best[1] if best else None


class Voices:
    """Answers of the speakers; remembers how often each command was asked."""

    def __init__(self, settings):
        self.settings = settings
        self.asked = {}  # command index -> times asked
        self.run = None

    def new_run(self, run):
        """A new task run starts with the first variant of each command again."""
        if run != self.run:
            self.run = run
            self.asked.clear()

    def answer(self, question, speaker):
        """What `speaker` says to the robot's `question`, or None."""
        if speaker is None:
            return None
        text = " ".join(re.sub(r"[^a-z0-9' ]+", " ", question.lower()).split())
        if speaker.operator and re.search(r"\bcommands?\b", text):
            index = next(
                (i for word, i in ORDINALS.items() if re.search(rf"\b{word}\b", text)),
                None,
            )
            commands = self.settings["commands"]
            if index is None or index >= len(commands):
                return None
            asked = self.asked.get(index, 0)
            self.asked[index] = asked + 1
            variants = commands[index]
            return variants[min(asked, len(variants) - 1)]
        name = re.search(r"\bare you ([a-z]+)\b", text)
        if name and speaker.name:
            if name.group(1) == speaker.name.lower():
                return "Yes."
            return f"No, my name is {speaker.name}."
        if re.search(r"\b(your name|who are you)\b", text) and speaker.name:
            return f"My name is {speaker.name}."
        if speaker.operator and re.search(
            r"\b(yes or no|is that (correct|right)|did you say)\b", text
        ):
            return self.settings["confirm"]
        return None
