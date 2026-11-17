"""Plan three GPSR commands: steps per command, then one interleaved plan. No ROS.

1. Each command becomes command steps [{"action", "params"}] (no memory keys):
   from the generator's exact phrasing by aha_tasks.gpsr_grammar, else by the
   LLM. The LLM gets a system prompt with the arena's words and the actions,
   few-shot examples made by the grammar parser, and a JSON schema built from
   the task definition and vocabulary (response_format json_schema: every
   action with its allowed param values). Invalid steps are sent back with the
   errors for another try, within a time budget.
2. `complete_command` makes plan steps of them (aha_tasks.plan): navigation
   before acting at a place, memory keys of answers (`cmd<N>.count`,
   `cmd<N>.info`) and found objects (`cmd<N>.object`), mock states, and the
   checks of one held object at a time and finding a person before acting on
   them.
3. `interleave` orders the commands' blocks (a navigation and what follows it
   there) to shorten the travel (straight lines between locations), keeping
   each command's order and holding at most one object.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import functools
import io
import json
import math
import time

from aha_tasks.gpsr_grammar import Parser, ParseError
from aha_tasks.gpsr_vocabulary import (
    COMPARISONS,
    GESTURE_PERSONS,
    GESTURE_PERSONS_PLURAL,
    OPERATOR,
    PERSON_INFO,
    POSE_PERSONS,
    POSE_PERSONS_PLURAL,
    TALK,
    article,
    clothing_phrases,
    spoken,
)
from aha_tasks.core.skill import resolve
from aha_tasks.plan import NOT_PLANNABLE, _validate_step
from aha_tasks.skills import LOCATION

HOLDING = "robot.holding"
MAX_COMMAND_STEPS = 12
# s of LLM time per request; a request never outlasts the planner's budget.
REQUEST_TIMEOUT = 45.0
SWITCH_COST = 0.01  # m; ties keep a command's blocks together
LOCATED = ("find_object", "find_described_person", "count", "count_people", "describe")
# Steps that act on the person found before (else they would run wherever the
# robot happens to be, to whoever is there).
NEEDS_PERSON = ("follow_person", "guide_person", "answer_question", "say")
ANSWERS = {"count": "count", "count_people": "count", "describe": "info"}

DESCRIPTIONS = {
    "navigate": "go to a location",
    "find_object": "look for an object (name or category) in a room or at a location",
    "pick_object": "pick up an object from a placement location (surface); "
    "leave surface out right after find_object",
    "place_object": "put the held object on a placement location (surface); "
    "'the trash' is {trash}",
    "deliver_object": "hand the held object to a person; person 'operator' is me",
    "find_described_person": "find a person by gesture, pose, clothing or name "
    "in a room or at a location, and approach them; with info alone, anyone there "
    "whose name, pose or gesture is asked for",
    "say": 'say a sentence, e.g. a greeting "Hello, I am AhaRobot."',
    "answer_question": "tell the person in front of the robot one of the talk topics",
    "count": "count objects of a category on a placement location",
    "count_people": "count the people (with a gesture, pose or clothing) in a room "
    "or at a location",
    "describe": "find out the biggest/smallest/... object, or object of a "
    "category, on a placement location",
    "tell_info": "tell the answer of the last count, count_people, describe or "
    "find_described_person with info to a person; person 'operator' is me; "
    "location is where another person is",
    "follow_person": "follow the found person (to the location if given)",
    "guide_person": "guide the found person to a location",
}

RULES = """\
Rules:
1. Navigate to a place before acting there: before find_object, \
find_described_person, count, count_people and describe go to their location; \
before pick_object and place_object go to the surface; before deliver_object or \
tell_info to the operator go to {operator}.
2. Find a person with find_described_person before giving them an object, \
following or guiding them. 'them' is the person found last. The operator is \
not searched for.
3. The robot holds one object at a time: pick_object, then place_object or \
deliver_object. pick_object takes the object from the placement location the \
command names; if the command names a room or another location, use find_object \
there and then pick_object without surface.
4. A question to me ('tell me how many ...', 'tell me what ...', 'tell me the \
name of ...') ends with navigate to {operator} and tell_info to the operator.
5. Use only the listed words and phrases. Map synonyms and misheard words to the \
closest listed one (coke -> cola, 'the person who waves' -> 'waving person').
6. guide_person: the robot leads the person (guide, escort, lead, take them to). \
follow_person: the robot goes after the person (follow).
7. Keep the order of the command. Output only the steps of this one command."""

# (example, canonical phrasing whose parse gives its steps; None: the example is)
EXAMPLES = (
    ("Bring me a cola from the kitchen table", None),
    (
        "Could you grab the lemon that is on the side tables and hand it to me?",
        "Bring me a lemon from the side tables",
    ),
    ("Tell me how many drinks there are on the desk", None),
    (
        "How many snacks are on the shelf?",
        "Tell me how many snacks there are on the shelf",
    ),
    ("Say the time to the person raising their left arm in the bedroom", None),
    (
        "Find an apple in the living room then take it and put it on the bed",
        None,
    ),
    (
        "Go to the office, look for the person who is sitting, and follow them to "
        "the kitchen.",
        "Go to the office then find the sitting person and follow them to the kitchen",
    ),
    (
        "Get a sponge from the shelf and deliver it to the waving person in the "
        "living room",
        None,
    ),
    ("Tell me the gesture of the person at the sofa", None),
    ("Greet Jane in the kitchen and guide them to the bed", None),
    ("Tell me how many people in the office are wearing white shirts", None),
    ("Tell the name of the person at the desk to the person at the bed", None),
    ("Take a snack from the shelf and throw it in the trash", None),
    ("Escort the person wearing a red coat from the entrance to the living room", None),
    ("Meet Robin at the chairs then find them in the bedroom", None),
    (
        "What is the heaviest fruit on the kitchen table?",
        "Tell me what is the heaviest fruit on the kitchen table",
    ),
)


class PlanError(ValueError):
    pass


# --- Schema and prompt ---------------------------------------------------------------


def _object(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _enum(values):
    return {"type": "string", "enum": list(dict.fromkeys(values))}


def plannable(spec):
    return [a for a in spec.skills if a not in NOT_PLANNABLE and a in DESCRIPTIONS]


def param_variants(vocabulary):
    """{action: [params schema, ...]} of the command steps."""
    v = vocabulary
    every = list(v.locations) + [v.operator]
    area = _enum(v.locations)
    placement = _enum(v.placements)
    objects = _enum(list(v.objects) + list(v.singular_categories))
    text = {"type": "string"}
    who = {
        "gesture": _enum(GESTURE_PERSONS),
        "pose": _enum(POSE_PERSONS),
        "clothing": _enum(clothing_phrases()),
        "name": _enum(v.names),
        "info": _enum(PERSON_INFO),
    }
    who_plural = {
        "gesture": _enum(GESTURE_PERSONS_PLURAL),
        "pose": _enum(POSE_PERSONS_PLURAL),
        "clothing": _enum(clothing_phrases(plural=True)),
    }
    described = [f"{c} object" for c in COMPARISONS]
    described += [f"{c} {s}" for c in COMPARISONS for s in v.singular_categories]
    return {
        "navigate": [_object({"location": _enum(every)}, ["location"])],
        "find_object": [
            _object({"object": objects, "location": area}, ["object", "location"])
        ],
        "pick_object": [
            _object({"object": objects, "surface": placement}, ["object", "surface"]),
            _object({"object": objects}, ["object"]),
        ],
        "place_object": [
            _object(
                {"object": objects, "surface": _enum(list(v.placements) + [v.trash])},
                ["object", "surface"],
            )
        ],
        "deliver_object": [
            _object({"object": objects, "person": text}, ["object", "person"])
        ],
        "find_described_person": [
            _object({key: schema, "location": area}, [key, "location"])
            for key, schema in who.items()
        ],
        "say": [_object({"text": text}, ["text"])],
        "answer_question": [_object({"question": _enum(TALK)}, ["question"])],
        "count": [
            _object(
                {
                    "target": _enum(list(v.plural_categories) + list(v.objects)),
                    "location": area,
                },
                ["target", "location"],
            )
        ],
        "count_people": [
            _object({key: schema, "location": area}, [key, "location"])
            for key, schema in who_plural.items()
        ]
        + [_object({"location": area}, ["location"])],
        "describe": [
            _object(
                {"target": _enum(described), "location": area}, ["target", "location"]
            )
        ],
        "tell_info": [
            _object({"person": text}, ["person"]),
            _object({"person": text, "location": area}, ["person", "location"]),
        ],
        "follow_person": [
            _object({"person": text, "location": area}, ["person", "location"]),
            _object({"person": text}, ["person"]),
        ],
        "guide_person": [
            _object({"person": text, "location": _enum(every)}, ["person", "location"])
        ],
    }


def command_schema(vocabulary, actions):
    """JSON schema of {"steps": [...]} for one command."""
    variants = param_variants(vocabulary)
    steps = [
        _object({"action": {"const": action}, "params": params}, ["action", "params"])
        for action in actions
        for params in variants[action]
    ]
    return _object(
        {
            "steps": {
                "type": "array",
                "items": {"anyOf": steps},
                "minItems": 1,
                "maxItems": MAX_COMMAND_STEPS,
            }
        },
        ["steps"],
    )


def _listing(words):
    return ", ".join(words)


def system_prompt(vocabulary, actions):
    v = vocabulary
    variants = param_variants(v)
    lines = [
        "You turn one spoken command for a home service robot into the robot's "
        "steps as JSON. The command follows the RoboCup@Home GPSR command "
        "generator but may be rephrased, and the speech recognizer may have "
        "misheard words.",
        "",
        "Arena (location labels):",
        f"- Rooms: {_listing(v.rooms)}",
        f"- Placement locations: {_listing(v.placements)}",
        f"- Other locations: {_listing(v.beacons)}",
        f"- The operator ('me') waits at {v.operator}; the robot starts there.",
        f"- Names of people: {_listing(v.names)}",
        "Objects by category (singular / plural):",
    ]
    by_category = {}
    for label, name in v.object_categories.items():
        by_category.setdefault(name, []).append(label)
    for category in v.categories:
        lines.append(
            f"- {category.singular} / {category.plural}: "
            f"{_listing(by_category.get(category.name, []))}"
        )
    lines += [
        f"Gestures: {_listing(GESTURE_PERSONS)} (plural: persons ...)",
        f"Poses: {_listing(POSE_PERSONS)}",
        "Clothing: person wearing a <colour> <garment> (plural: people wearing "
        "<colour> <garments>)",
        f"Talk topics: {_listing(TALK)}",
        f"Comparisons: {_listing(COMPARISONS)}",
        "",
        "Actions (params; '|' separates alternatives, '?' is optional):",
    ]
    for action in actions:
        forms = []
        for schema in variants[action]:
            keys = list(schema["properties"])
            forms.append(", ".join(keys))
        description = DESCRIPTIONS[action].format(trash=v.trash)
        lines.append(f"- {action} {{{' | '.join(forms)}}}: {description}")
    lines += ["", RULES.format(operator=v.operator)]
    return "\n".join(lines)


def few_shot(vocabulary, parser=None):
    """User / assistant messages of the examples that fit the vocabulary."""
    parser = parser or Parser(vocabulary)
    messages = []
    for text, canonical in EXAMPLES:
        try:
            steps = parser.parse(canonical or text)
        except ParseError:
            continue  # a word the arena does not have
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps({"steps": steps})})
    return messages


# --- Command steps -> plan steps -----------------------------------------------------


def required_place(action, params, vocabulary):
    """The location label the robot must be at for a step, or None."""
    if action in LOCATED:
        return params.get("location")
    if action in ("pick_object", "place_object"):
        return params.get("surface")
    if action in ("deliver_object", "tell_info") and params.get("person") == OPERATOR:
        return vocabulary.operator
    if action == "tell_info":
        return params.get("location")
    return None


def asks_info(spec, params):
    """True if the task's find_described_person takes `info` (person info to tell).

    Without it, the person's name, pose or gesture is a describe target.
    """
    skill = resolve(spec.skills.get("find_described_person"))
    if skill is None:
        return True
    try:
        skill.validate(params, spec)
    except ValueError:
        return False
    return True


def object_phrase(step):
    """Spoken words of a step's object (a found object's key gives its words)."""
    if step["action"] == "pick_object":
        return step.get("mock", {}).get(HOLDING) or step["params"]["object"]
    return step["params"].get("object", "")


def complete_command(steps, number, vocabulary, spec):
    """(plan steps, errors) of one command's steps.

    Adds missing navigation, memory keys and mock states, drops navigation to
    where the robot already is (the place of its last step), and checks the
    order of picking, placing, finding and telling. The person info of a
    find_described_person goes to `cmd<N>.person.<info>` (role `cmd<N>.person`),
    so that another command's search in between does not replace it.
    """
    actions = set(plannable(spec))
    errors, out = [], []
    area = None  # where the robot is
    holding = found_object = last_answer = None
    found_person = False
    answers = {}

    def add(action, params, mock=None):
        step = {"command": number, "action": action, "params": params}
        if mock:
            step["mock"] = mock
        if action == "navigate" and out and out[-1]["action"] == "navigate":
            out.pop()  # nothing was done there
        out.append(step)

    if not isinstance(steps, list) or not steps:
        return [], ["needs at least one step"]
    for index, raw in enumerate(steps, 1):
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("action"), str)
            or not isinstance(raw.get("params"), dict)
        ):
            errors.append(f"step {index}: needs an action and params: {raw!r}")
            continue
        action, params = raw["action"], dict(raw["params"])
        if action not in actions:
            errors.append(f"step {index}: unknown action {action!r}")
            continue
        place = required_place(action, params, vocabulary)
        if place is not None and place != area and action != "navigate":
            add("navigate", {"location": place})
            area = place
            found_person = False
        mock = {}
        if action == "navigate":
            location = params.get("location")
            if location == area:  # already there
                continue
            add(action, params)
            area = location
            found_person = False
            continue
        if action == "find_object":
            key = f"cmd{number}.object"
            params["output"] = key
            found_object = (params.get("object"), key)
        elif action == "find_described_person" and "info" in params:
            info, role = params["info"], f"cmd{number}.person"
            params["role"] = role
            last_answer = f"{role}.{info}"
            found_person = True
            if not asks_info(spec, params):
                action = "describe"
                last_answer = f"cmd{number}.info"
                params = {
                    "target": f"{info} of the person",
                    "location": params.get("location"),
                    "output": last_answer,
                }
                found_person = False
        elif action == "find_described_person":
            found_person = True
        elif action == "pick_object":
            phrase = params.get("object")
            if holding is not None:
                errors.append(
                    f"step {index}: the robot still holds the {holding}; place or "
                    "deliver it before picking another object"
                )
            if "surface" not in params:
                if found_object is None:
                    errors.append(
                        f"step {index}: pick_object needs a surface or a find_object "
                        "step before it"
                    )
                else:
                    params["object"] = found_object[1]
            holding = phrase
            mock[HOLDING] = phrase
        elif action in ("place_object", "deliver_object"):
            if holding is None:
                errors.append(f"step {index}: {action} without a held object")
            elif action == "deliver_object" and params.get("person") != OPERATOR:
                if not found_person:
                    errors.append(
                        f"step {index}: find the person with find_described_person "
                        "before deliver_object"
                    )
            holding = None
            mock[HOLDING] = None
        elif action in ANSWERS:
            base = f"cmd{number}.{ANSWERS[action]}"
            answers[base] = answers.get(base, 0) + 1
            key = base if answers[base] == 1 else f"{base}{answers[base]}"
            params["output"] = key
            last_answer = key
        elif action == "tell_info":
            if last_answer is None:
                errors.append(
                    f"step {index}: tell_info needs a count, count_people, "
                    "describe or find_described_person with info before it"
                )
            if (
                params.get("person") != OPERATOR
                and "location" not in params
                and not found_person
            ):
                errors.append(
                    f"step {index}: tell_info to {params.get('person')!r} needs "
                    f"person {OPERATOR!r}, a location, or find_described_person "
                    "before it"
                )
            params = {"key": last_answer or f"cmd{number}.info", **params}
        elif action in NEEDS_PERSON and not found_person:
            errors.append(
                f"step {index}: find the person with find_described_person before "
                f"{action}"
            )
        if action in ("follow_person", "guide_person"):
            area = params.get("location")
            if area is not None:
                mock[LOCATION] = area
        add(action, params, mock)
    if holding is not None:
        errors.append(f"the {holding} is still held at the end; place or deliver it")
    if not any(step["action"] != "navigate" for step in out):
        errors.append("needs a step other than navigate")
    for index, step in enumerate(out, 1):
        try:
            _validate_step(step, spec)
        except ValueError as error:
            errors.append(f"{step['action']} {step['params']}: {error}")
    return out, errors


# --- LLM ------------------------------------------------------------------------------


@dataclass
class CommandPlan:
    number: int
    text: str
    steps: list = field(default_factory=list)  # plan steps, empty if not planned
    planner: str = ""  # "grammar" | "llm" | "" (failed)
    error: str = ""
    attempts: list = field(default_factory=list)  # (seconds, errors) per LLM try
    dropped: bool = False  # planned, but left out: the plan did not fit the slots

    def info(self, vocabulary):
        return {
            "command": self.number,
            "text": self.text,
            "planner": self.planner,
            "summary": command_summary(self.steps, vocabulary) if self.steps else "",
            "error": self.error,
            "dropped": self.dropped,
        }


def parse_reply(content):
    """Command steps of the LLM's JSON content."""
    try:
        data = json.loads(content)
    except ValueError as error:
        raise PlanError(f"not JSON: {error}") from None
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise PlanError('expected {"steps": [...]}')
    return data["steps"]


class Planner:
    """Plans commands with the grammar parser and an LLM (`chat`, see aha_tasks.llm).

    `mode`: "auto" tries the parser first, "llm" or "grammar" uses only one.
    """

    def __init__(
        self, vocabulary, spec, chat=None, mode="auto", attempts=3, thinking=False
    ):
        self.vocabulary, self.spec, self.chat = vocabulary, spec, chat
        self.mode, self.attempts, self.thinking = mode, attempts, thinking
        self.parser = Parser(vocabulary)
        self.actions = plannable(spec)
        self.schema = command_schema(vocabulary, self.actions)
        self.messages = [
            {"role": "system", "content": system_prompt(vocabulary, self.actions)}
        ] + few_shot(vocabulary, self.parser)

    def plan_command(self, number, text, deadline, cancelled=None):
        result = CommandPlan(number, text)
        if not text or not text.strip():
            result.error = "no command text"
            return result
        if self.mode in ("auto", "grammar"):
            try:
                steps, errors = complete_command(
                    self.parser.parse(text), number, self.vocabulary, self.spec
                )
                if not errors:
                    result.steps, result.planner = steps, "grammar"
                    return result
                result.error = "; ".join(errors)
            except ParseError as error:
                result.error = str(error)
        if self.mode == "grammar":
            return result
        if self.chat is None:
            result.error = "no LLM"
            return result
        self._ask_llm(result, deadline, cancelled)
        return result

    def _ask_llm(self, result, deadline, cancelled):
        messages = self.messages + [{"role": "user", "content": result.text}]
        for _ in range(self.attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0 or (cancelled is not None and cancelled.is_set()):
                result.error = result.error or "out of time"
                return
            try:
                reply = self.chat(
                    messages,
                    schema=self.schema,
                    thinking=self.thinking,
                    max_tokens=8192 if self.thinking else 2048,
                    timeout=min(remaining, REQUEST_TIMEOUT),
                )
            except Exception as error:  # unreachable server, timeout
                result.error = f"LLM: {error}"
                result.attempts.append((None, [result.error]))
                return
            try:
                steps, errors = complete_command(
                    parse_reply(reply.content),
                    result.number,
                    self.vocabulary,
                    self.spec,
                )
            except PlanError as error:
                steps, errors = [], [str(error)]
            result.attempts.append((reply.seconds, errors))
            if not errors:
                result.steps, result.planner, result.error = steps, "llm", ""
                return
            result.error = "; ".join(errors)
            messages = messages + [
                {"role": "assistant", "content": reply.content},
                {
                    "role": "user",
                    "content": "These steps are not valid: "
                    + " ".join(f"({i}) {e}" for i, e in enumerate(errors, 1))
                    + " Reply with the corrected steps of the same command.",
                },
            ]

    def plan(self, texts, timeout, cancelled=None, start=None):
        """(plan, [CommandPlan]) of the commands {number: text}; the LLM runs in parallel."""
        deadline = time.monotonic() + timeout
        with ThreadPoolExecutor(max_workers=max(1, len(texts))) as pool:
            futures = [
                pool.submit(self.plan_command, number, text, deadline, cancelled)
                for number, text in sorted(texts.items())
            ]
            results = [future.result() for future in futures]
        commands = {r.number: r.steps for r in results if r.steps}
        plan = interleave(
            commands, self.spec, start or self.vocabulary.operator, self.vocabulary
        )
        capacity = sum(1 for s in self.spec.subgoals if s.skill == "run_plan_step")
        while len(plan) > capacity and commands:
            longest = max(commands, key=lambda n: (len(commands[n]), n))
            dropped = next(r for r in results if r.number == longest)
            dropped.error = f"the plan does not fit {capacity} steps"
            dropped.steps, dropped.planner, dropped.dropped = [], "", True
            del commands[longest]
            plan = interleave(
                commands, self.spec, start or self.vocabulary.operator, self.vocabulary
            )
        return plan, results


# --- Interleaving ----------------------------------------------------------------------


@dataclass
class Block:
    command: int
    steps: list
    start: str | None  # navigation target at the beginning
    end: str | None  # where the robot is after the block
    holds: bool  # the command holds an object after the block
    picks: bool = False  # the block picks an object (even if it then gives it away)


def blocks_of(steps):
    """A command's steps split before each navigation."""
    blocks, holding = [], False
    for step in steps:
        if step["action"] == "navigate" or not blocks:
            start = step["params"]["location"] if step["action"] == "navigate" else None
            blocks.append(Block(step["command"], [], start, start, holding))
        block = blocks[-1]
        block.steps.append(step)
        if step["action"] == "pick_object":
            holding = True
            block.picks = True
        elif step["action"] in ("place_object", "deliver_object"):
            holding = False
        elif step["action"] in ("follow_person", "guide_person"):
            location = step["params"].get("location")
            block.end = location if location is not None else block.end
        block.holds = holding
    return blocks


def _distance(spec, a, b):
    if a is None or b is None or a == b:
        return 0.0
    (xa, ya, _), (xb, yb, _) = spec.locations[a], spec.locations[b]
    return math.hypot(xa - xb, ya - yb)


def travel(blocks, spec, start, finish):
    """m of straight lines from `start` through the blocks to `finish`."""
    total, here = 0.0, start
    for block in blocks:
        total += _distance(spec, here, block.start)
        here = block.end or block.start or here
    return total + _distance(spec, here, finish)


def order_blocks(commands, spec, start, finish):
    """Blocks of {number: [Block]} in the order of the least travel.

    Each command keeps its order, and at most one command holds an object at a
    time. Ties keep the commands' blocks together, in command order.
    """
    numbers = sorted(commands)
    sequences = [commands[n] for n in numbers]

    def holders(position):
        return sum(1 for seq, i in zip(sequences, position) if i and seq[i - 1].holds)

    @functools.lru_cache(maxsize=None)
    def best(position, here, last):
        if all(i == len(seq) for seq, i in zip(sequences, position)):
            return _distance(spec, here, finish), ()
        result = (math.inf, ())
        for c, seq in enumerate(sequences):
            i = position[c]
            if i == len(seq):
                continue
            block = seq[i]
            after = position[:c] + (i + 1,) + position[c + 1 :]
            if holders(after) > 1:
                continue
            if block.picks and holders(position[:c] + (0,) + position[c + 1 :]):
                continue  # another command holds an object now
            cost = _distance(spec, here, block.start)
            if last is not None and last != c:
                cost += SWITCH_COST
            rest, order = best(after, block.end or block.start or here, c)
            if cost + rest < result[0] - 1e-9:
                result = (cost + rest, (c,) + order)
        return result

    cost, order = best(tuple(0 for _ in sequences), start, None)
    if math.isinf(cost):
        raise PlanError("the commands cannot be ordered holding one object at a time")
    taken, blocks = [0] * len(sequences), []
    for c in order:
        blocks.append(sequences[c][taken[c]])
        taken[c] += 1
    best.cache_clear()
    return blocks


def join(blocks, start=None):
    """Plan steps of ordered blocks, without navigation to where the robot is."""
    plan, area = [], start
    for block in blocks:
        for step in block.steps:
            action = step["action"]
            if action == "navigate":
                if step["params"]["location"] == area:
                    continue
                area = step["params"]["location"]
            elif action in ("follow_person", "guide_person"):
                area = step["params"].get("location")
            plan.append(step)
    return plan


def interleave(commands, spec, start, vocabulary=None):
    """One plan of {number: plan steps}, ordered for the least travel."""
    finish = vocabulary.operator if vocabulary is not None else start
    blocks = {n: blocks_of(steps) for n, steps in commands.items() if steps}
    if not blocks:
        return []
    return join(order_blocks(blocks, spec, start, finish), start)


def sequential(commands):
    return [step for n in sorted(commands) for step in commands[n]]


def plan_travel(plan, spec, start, finish):
    """m of straight lines of a plan's navigation, from start to finish."""
    total, here = 0.0, start
    for step in plan:
        location = None
        if step["action"] == "navigate":
            location = step["params"]["location"]
        elif step["action"] in ("follow_person", "guide_person"):
            location = step["params"].get("location")
        if location is not None:
            total += _distance(spec, here, location)
            here = location
    return total + _distance(spec, here, finish)


def interleaved(plan):
    """True if the plan leaves a command and comes back to it later."""
    seen, last = set(), None
    for step in plan:
        number = step["command"]
        if number != last:
            if number in seen:
                return True
            seen.add(number)
            last = number
    return False


# --- Words ---------------------------------------------------------------------------------


def _where(label, vocabulary, on=False):
    if label in vocabulary.rooms:
        preposition = "in"
    else:
        preposition = "on" if on and label in vocabulary.placements else "at"
    return f"{preposition} the {spoken(label)}"


def _person_words(params):
    if "name" in params:
        return params["name"]
    for key in ("gesture", "pose", "clothing"):
        if key in params:
            return f"the {params[key]}"
    return "the people"


def person_words(person):
    """'me', a name, or 'the <description>'."""
    if not person:
        return "them"
    if person == OPERATOR:
        return "me"
    if person[:1].isupper() or person.startswith("the "):
        return person
    return f"the {person}"


TALK_WORDS = {
    "something about yourself": "something about myself",
    "your teams name": "our team's name",
    "your teams country": "our team's country",
    "your teams affiliation": "our team's affiliation",
    "what day is today": "what day it is today",
    "what day is tomorrow": "what day it is tomorrow",
}


def step_phrase(step, vocabulary):
    """English words of a plan step: 'pick up the cola from the kitchen table'."""
    action, p = step["action"], step["params"]
    person = p.get("person", "")
    to = person_words(person)
    if action == "navigate":
        return f"go to the {spoken(p['location'])}"
    if action == "find_object":
        obj = p["object"]
        return f"find {article(obj)} {obj} {_where(p['location'], vocabulary)}"
    if action == "pick_object":
        words = f"pick up the {object_phrase(step)}"
        return words + (f" from the {spoken(p['surface'])}" if "surface" in p else "")
    if action == "place_object":
        if p["surface"] == vocabulary.trash:
            return f"throw the {p['object']} in the trash"
        return f"put the {p['object']} on the {spoken(p['surface'])}"
    if action == "deliver_object":
        return f"give the {p['object']} to {to}"
    if action == "find_described_person":
        where = _where(p["location"], vocabulary)
        if set(p) & {"name", "gesture", "pose", "clothing", "description"}:
            return f"find {_person_words(p)} {where}"
        return f"find out the {p.get('info', 'name')} of the person {where}"
    if action == "say":
        return f'say "{p["text"]}"'
    if action == "answer_question":
        return f"tell {TALK_WORDS.get(p['question'], p['question'])}"
    if action == "count":
        return f"count the {p['target']} {_where(p['location'], vocabulary, on=True)}"
    if action == "count_people":
        return f"count {_person_words(p)} {_where(p['location'], vocabulary)}"
    if action == "describe":
        return (
            f"find out the {p['target']} {_where(p['location'], vocabulary, on=True)}"
        )
    if action == "tell_info":
        return (
            "tell you the answer" if person == OPERATOR else f"tell the answer to {to}"
        )
    if action == "follow_person":
        words = f"follow {to}"
        return words + (f" to the {spoken(p['location'])}" if "location" in p else "")
    if action == "guide_person":
        return f"guide {to} to the {spoken(p['location'])}"
    return f"{action.replace('_', ' ')} {p}"


def _join(phrases):
    if len(phrases) <= 1:
        return "".join(phrases)
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def command_summary(steps, vocabulary):
    """A command's steps without navigation: 'pick up the cola ... and give it ...'."""
    phrases = [step_phrase(s, vocabulary) for s in steps if s["action"] != "navigate"]
    return _join(phrases or [step_phrase(s, vocabulary) for s in steps])


def plan_speech(plan, vocabulary):
    """Sentences that say the plan in its order, a command at a time."""
    segments = []
    for step in plan:
        if not segments or segments[-1][0] != step["command"]:
            segments.append((step["command"], []))
        if step["action"] != "navigate":
            segments[-1][1].append(step_phrase(step, vocabulary))
    segments = [(n, phrases) for n, phrases in segments if phrases]
    sentences = []
    for i, (number, phrases) in enumerate(segments):
        lead = "First" if i == 0 else "Finally" if i == len(segments) - 1 else "Then"
        sentences.append(f"{lead}, for command {number}, I will {_join(phrases)}.")
    return " ".join(sentences)


# --- Picture -------------------------------------------------------------------------------

COLOURS = ((31, 119, 180), (214, 39, 40), (44, 160, 44))  # command 1, 2, 3
SIZE = (1280, 720)


def _font(size):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_plan(plan, vocabulary, title="My plan"):
    """PNG bytes of the numbered plan steps, coloured by command."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    rows = max(1, len(plan))
    line = min(44, (SIZE[1] - 90) // rows)
    font, heading = _font(max(14, int(line * 0.72))), _font(40)
    draw.text((30, 18), f"{title} ({len(plan)} steps)", fill=(0, 0, 0), font=heading)
    x = SIZE[0] - 30
    for number in sorted({s["command"] for s in plan}, reverse=True):
        label = f"Command {number}"
        width = draw.textlength(label, font=_font(28))
        x -= width + 50
        colour = COLOURS[(number - 1) % len(COLOURS)]
        draw.rectangle((x, 28, x + 30, 52), fill=colour)
        draw.text((x + 38, 24), label, fill=(0, 0, 0), font=_font(28))
    for i, step in enumerate(plan):
        y = 80 + i * line
        colour = COLOURS[(step["command"] - 1) % len(COLOURS)]
        draw.rectangle((30, y + 4, 40, y + line - 4), fill=colour)
        text = f"{i + 1:>2}. {step_phrase(step, vocabulary)}"
        draw.text((56, y), text, fill=colour, font=font)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


# --- Report -------------------------------------------------------------------------------


def report_sentences(plan, commands, outcomes, memory):
    """What was done per command.

    `commands`: [{"command", "summary", "error"}] (memory "plan.commands");
    `outcomes`: {plan index: "done" | "failed" | "mocked" | None (unknown)}.
    """
    sentences = []
    for info in commands:
        number = info["command"]
        if info.get("dropped"):
            sentences.append(
                f"Command {number}: I understood it, but it did not fit my plan."
            )
            continue
        if not info.get("summary"):
            sentences.append(f"Command {number}: I did not understand it.")
            continue
        indices = [i for i, step in enumerate(plan) if step["command"] == number]
        states = [outcomes.get(i) for i in indices]
        sentence = f"Command {number}, {info['summary']}"
        if states and all(state == "done" for state in states):
            sentence += ": done."
        elif any(state in ("failed", "mocked") for state in states):
            done = sum(1 for state in states if state == "done")
            sentence += f": not finished, {done} of {len(states)} steps done."
        else:  # no task status
            sentence += "."
        for i in indices:
            # Only a step that ran: a mocked one leaves its mock value.
            key = answer_key(plan[i])
            if outcomes.get(i) == "done" and key is not None and key in memory:
                answer = memory.get(f"{key}.answer") or f"The answer was {memory[key]}."
                sentence += f" {answer}"
        sentences.append(sentence)
    return sentences


def answer_key(step):
    """The memory key of the answer a plan step finds out, or None."""
    params = step["params"]
    if step["action"] in ANSWERS:
        return params.get("output")
    if step["action"] == "find_described_person" and "info" in params:
        return f"{params.get('role', 'person')}.{params['info']}"
    return None


def step_outcomes(status, plan):
    """{plan index: outcome} from the task status (/aha/tasks/status JSON)."""
    outcomes = {}
    subgoals = (status or {}).get("subgoals", [])
    slots = [s for s in subgoals if s.get("skill") == "run_plan_step"]
    # The slots run the plan's indices 0, 1, ... in order (aha_tasks.plan).
    for index, subgoal in enumerate(slots[: len(plan)]):
        state, message = subgoal.get("status"), subgoal.get("message") or ""
        if state == "succeeded":
            outcomes[index] = "mocked" if "not implemented" in message else "done"
        elif state in ("failed", "skipped", "cancelled", "unsupported"):
            outcomes[index] = "failed"
    return outcomes
