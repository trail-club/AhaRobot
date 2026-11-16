"""GPSR skills that talk to a person: answer_question and tell_info.

The sentences come from aha_tasks.conversation, and the robot says them with
aha_tasks.interaction (a caption, and speech with a TTS). tell_info first
turns to the person it tells: the operator at the instruction point, the
person seen at a location, or the person a find_described_person step put in
memory (`<role>.position`). Without person perception, or when nobody is seen,
it speaks where it is.
"""

import itertools
import math
from types import SimpleNamespace

from aha_tasks.conversation import (
    KNOWN_QUESTIONS,
    UTC_OFFSET_KEY,
    answer,
    check_utc_offset,
    facts_of,
    local_now,
    missing_facts,
    question_topic,
    value_sentence,
)
from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.interaction import Interaction, text
from aha_tasks.people import (
    People,
    approach_pose,
    check_keys,
    check_name,
    planar,
    positive,
    robot_xy,
)
from aha_tasks.skills import LOCATION, go_to, resolve_pose

INSTRUCTION_POINT = "instruction_point"
OPERATOR_WORDS = {"operator", "the operator", "me"}
FACE_PANS = (0.0, 0.6, -0.6)  # rad; where the head looks for a person to face
FACE_DWELL = 2.0  # s (wait clock) at each pan
FACE_TIMEOUT = 6.0  # s (wait clock) to see the person
FACE_RADIUS = 2.0  # m around the location for the person to face there
NEAR = 1.5  # m; a closer person is only turned to, a farther one approached
REACQUIRE = 0.7  # m from the remembered position for a new track of the person
HEAD_ABOVE_CENTRE = 0.5  # m; where to look while the person is not seen


def is_operator(person):
    return " ".join(person.lower().split()) in OPERATOR_WORDS


class AnswerQuestion(Skill):
    """Say the answer to a question about the robot, its team, the time or the date.

    params: {"question": "<the generator's {talk}, e.g. 'the time', or a
    rephrasing>"}. Topics (aha_tasks.conversation.question_topic): something
    about yourself, the time, today, tomorrow, the day of the week or month,
    the team's name, country and affiliation. The facts are `gpsr.*` in
    memory (initial_state); the time is the wall clock at `gpsr.utc_offset`.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"question"})
        topic = question_topic(text(params, "question"))
        # An unknown question is unsupported (only this step is mocked); a
        # fact missing from the task definition is an error.
        if topic is not None:
            missing = missing_facts(topic, facts_of(spec.initial_state))
            if missing:
                raise ValueError(f"{', '.join(missing)} must be set in initial_state")
        check_utc_offset(spec.initial_state.get(UTC_OFFSET_KEY))

    def unsupported(self, params):
        if question_topic(params["question"]) is None:
            return (
                f"cannot answer {params['question']!r}; "
                f"known: {', '.join(KNOWN_QUESTIONS)}"
            )
        return None

    def ready(self):
        Interaction.of(self.runtime).find_tts()

    def execute(self, ctx):
        topic = question_topic(ctx.params["question"])
        if topic is None:
            return SkillResult(False, self.unsupported(ctx.params))
        try:
            sentence = answer(topic, facts_of(ctx.memory), local_now(ctx.memory))
        except ValueError as error:
            return SkillResult(False, str(error))
        ok, message = Interaction.of(self.runtime).say(sentence, ctx.cancelled)
        return SkillResult(ok, sentence if ok else f"{sentence} ({message})")


class TellInfo(Skill):
    """Tell a person what a memory key holds: `<key>.answer`, else its value phrased.

    params: {"key": "<memory key>", "person": "operator" | "<who>",
    "role": "person", "location": "<where the person is>", "distance": 1.0}.
    The robot first turns to the person:

    - "operator" (or "me"): it goes to the instruction point and faces the
      nearest person seen within FACE_RADIUS;
    - with `location`: it goes there and faces the nearest person seen within
      FACE_RADIUS;
    - otherwise the person of `<role>.track` / `.name` / `.position`
      (find_described_person, `role` "person" by default).

    A person farther than NEAR is approached to `distance`.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"key", "person"}, {"role", "location", "distance"})
        text(params, "key")
        text(params, "person")
        if "role" in params:
            check_name(params, "role")
        if "location" in params:
            if not isinstance(params["location"], str):
                raise ValueError(f"location must be a label: {params['location']!r}")
            resolve_pose(spec, params["location"])
        if is_operator(params["person"]) and INSTRUCTION_POINT not in spec.locations:
            raise ValueError(f"telling the operator needs the {INSTRUCTION_POINT}")
        positive(params, "distance", 1.0)

    @staticmethod
    def place(params):
        """The location to go to before facing the person, or None."""
        if is_operator(params["person"]):
            return INSTRUCTION_POINT
        return params.get("location")

    def expected(self, params):
        place = self.place(params)
        return {} if place is None else {LOCATION: place}

    def ready(self):
        Interaction.of(self.runtime).find_tts()

    def execute(self, ctx):
        params, memory = ctx.params, ctx.memory
        key = params["key"]
        sentence = memory.get(f"{key}.answer")
        if not isinstance(sentence, str) or not sentence.strip():
            if key not in memory:
                return SkillResult(False, f"Nothing to tell: {key} is not in memory")
            try:
                sentence = value_sentence(key, memory[key])
            except ValueError as error:
                return SkillResult(False, f"Nothing to tell: {error}")
        outputs = {}
        place = self.place(params)
        if place is not None:
            if memory.get(LOCATION) != place and self.runtime.navigation is not None:
                moved = go_to(self.runtime, place, ctx.cancelled)
                if not moved.success:
                    return SkillResult(
                        False, f"Could not go to {place}: {moved.message}"
                    )
            outputs[LOCATION] = place
            error = self.face_nearest(ctx, place, outputs)
        else:
            error = self.face_role(ctx, outputs)
        if error:
            # Turning to or approaching them failed: tell it from here.
            ctx.log(f"{error}: telling from here")
        if ctx.cancelled.is_set():
            return SkillResult(False, "Cancelled")
        ok, message = Interaction.of(self.runtime).say(sentence, ctx.cancelled)
        told = f"Told {params['person']}: {sentence}"
        return SkillResult(ok, told if ok else f"{told} ({message})", outputs)

    def people(self):
        """Person perception, or None without ROS (tests)."""
        runtime = self.runtime
        if getattr(runtime, "people", None) is None and runtime.node is None:
            return None
        return People.of(runtime)

    def face_nearest(self, ctx, place, outputs):
        """Face the nearest person seen at `place`; an error, or None."""
        people = self.people()
        if people is None:
            return None
        origin = tuple(resolve_pose(self.runtime.spec, place)[:2])
        pans = itertools.cycle(FACE_PANS)
        next_look = -math.inf

        def choose(message):
            nonlocal next_look
            if people.now() >= next_look:
                people.look_towards(next(pans))
                next_look = people.now() + FACE_DWELL
            visible = [p for p in message.people if p.visible]
            if message.header.frame_id == "map":
                near = [
                    (math.dist(planar(p.position), origin), p.id, p) for p in visible
                ]
                near = [item for item in near if item[0] <= FACE_RADIUS]
                return (message, min(near)[2]) if near else None
            if not visible:
                return None
            return message, min(
                visible, key=lambda p: math.hypot(p.position.x, p.position.y)
            )

        found, reason = people.wait_for(choose, FACE_TIMEOUT, ctx.cancelled)
        if found is None:
            people.look_towards(0.0)
            ctx.log(f"Nobody to face at {place}: {reason}")
            return None
        return self.face(ctx, people, *found, origin, outputs)

    def face_role(self, ctx, outputs):
        """Face the person of `<role>.*` in memory; an error, or None."""
        role = ctx.params.get("role", "person")
        memory = ctx.memory
        position = memory.get(f"{role}.position")
        frame = memory.get(f"{role}.frame")
        if not (isinstance(position, list) and len(position) == 3) or not frame:
            ctx.log(f"No {role}.position in memory: telling from here")
            return None
        track = memory.get(f"{role}.track")
        name = memory.get(f"{role}.name") or ""
        origin = robot_xy(self.runtime, memory)
        people = self.people()
        if people is not None:
            x, y, z = position
            people.look_at(frame, SimpleNamespace(x=x, y=y, z=z + HEAD_ABOVE_CENTRE))

            def choose(message):
                visible = [p for p in message.people if p.visible]
                for person in visible:
                    if person.id == track or (name and person.name == name):
                        return message, person
                if message.header.frame_id != frame:
                    return None
                near = [
                    (math.dist(planar(p.position), position[:2]), p.id, p)
                    for p in visible
                    if not p.name or p.name == name
                ]
                near = [item for item in near if item[0] < REACQUIRE]
                return (message, min(near)[2]) if near else None

            found, reason = people.wait_for(choose, FACE_TIMEOUT, ctx.cancelled)
            if found is not None:
                return self.face(ctx, people, *found, origin, outputs)
            ctx.log(f"{role} not seen ({reason}): turning to where they were")
        if frame != "map":
            return None
        return self.turn_to(ctx, position[:2], origin, outputs)

    def face(self, ctx, people, message, person, origin, outputs):
        people.follow(track=person.id, name=person.name)
        if message.header.frame_id != "map":
            return None
        return self.turn_to(ctx, planar(person.position), origin, outputs)

    def turn_to(self, ctx, target, origin, outputs):
        """Turn the base to `target` (map), approaching it from beyond NEAR."""
        navigation = self.runtime.navigation
        if navigation is None:
            return None
        if math.dist(origin, target) > NEAR:
            pose = approach_pose(origin, target, positive(ctx.params, "distance", 1.0))
        else:
            yaw = math.atan2(target[1] - origin[1], target[0] - origin[0])
            pose = [origin[0], origin[1], yaw]
        ok, message = navigation.navigate(pose, ctx.cancelled)
        if not ok:
            return f"Could not turn to {ctx.params['person']}: {message}"
        outputs[LOCATION] = [round(v, 3) for v in pose]
        return None
