"""GPSR plan steps about objects: find one, count them, tell the biggest / heaviest.

Commands of the RoboCup@Home command generator (rulebook 5.3): findObjInRoom
and findObj (`find_object`), countObjOnPlcmt (`count`), tellObjPropOnPlcmt and
tellCatPropOnPlcmt (`describe`). A step's `location` is a room or a furniture
location of the task definition's `rooms`, which also names the furniture
whose top is looked over (aha_tasks.objects, `Objects.scan`). The furniture of
a room is visited on the shortest straight-line path from the robot.

Names in commands are matched to the object list (aha_perception.object_list)
without regard to case, plurals, punctuation ("rubik's", "jell-o"), word order
or a container word ("tomato soup can", "can of cola"), and through the list's
`aliases` (other names of a label) and `groups` (e.g. "kitchen item"). People
("the name of the person") are not this module's: such steps are unsupported,
so a plan mocks them.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.interaction import Interaction
from aha_tasks.objects import Cancelled, Objects, check_margin, norm_label, record
from aha_tasks.people import check_keys, check_name
from aha_tasks.skills import LOCATION, go_to, resolve_pose

MARGIN = 0.1  # m beyond a furniture outline where objects still count as on it
ARTICLES = {"the", "a", "an", "some", "any", "all", "every", "each"}
# Words of a question around the subject: "what is the smallest item".
FILLER = ARTICLES | {"what", "which", "is", "are", "that", "there", "me", "tell"}
PREPOSITIONS = {"on", "in", "at", "from", "inside", "near"}
GENERIC = {"object", "item", "thing", "one", "anything", "something", "stuff"}
CONTAINERS = {
    "can",
    "box",
    "bottle",
    "carton",
    "pack",
    "packet",
    "bag",
    "jar",
    "tin",
    "pouch",
    "tube",
    "package",
}
PERSON_WORDS = {
    "person",
    "people",
    "man",
    "woman",
    "guy",
    "lady",
    "individual",
    "someone",
    "somebody",
    "human",
    "operator",
    "child",
    "kid",
    "boy",
    "girl",
    "guest",
    "visitor",
    "name",
    "pose",
    "posture",
    "gesture",
}
IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child"}
# Superlative -> (measure, largest value first).
SUPERLATIVES = {
    "biggest": ("volume", True),
    "largest": ("volume", True),
    "smallest": ("volume", False),
    "heaviest": ("weight", True),
    "lightest": ("weight", False),
    "thinnest": ("thickness", False),
}
# Other words for them.
SYNONYMS = {
    "bulkiest": "biggest",
    "hugest": "biggest",
    "greatest": "biggest",
    "most big": "biggest",
    "most large": "largest",
    "most bulky": "biggest",
    "most voluminous": "biggest",
    "tiniest": "smallest",
    "littlest": "smallest",
    "teeniest": "smallest",
    "least big": "smallest",
    "least large": "smallest",
    "most compact": "smallest",
    "most heavy": "heaviest",
    "least light": "heaviest",
    "weighs the most": "heaviest",
    "weigh the most": "heaviest",
    "least heavy": "lightest",
    "weighs the least": "lightest",
    "weigh the least": "lightest",
    "slimmest": "thinnest",
    "skinniest": "thinnest",
    "narrowest": "thinnest",
    "slenderest": "thinnest",
    "flattest": "thinnest",
    "most thin": "thinnest",
    "most slim": "thinnest",
    "most slender": "thinnest",
    "most narrow": "thinnest",
    "least thick": "thinnest",
}
# Words of a superlative or a synonym -> the superlative.
PHRASES = {(name,): name for name in SUPERLATIVES} | {
    tuple(phrase.split()): name for phrase, name in SYNONYMS.items()
}


# -- names -------------------------------------------------------------------


def words(text):
    """Lower-case words without punctuation or leading articles:
    "The Rubik's cube" -> ["rubiks", "cube"], "jell-o" -> ["jello"]."""
    text = str(text).lower().replace("’", "'")
    text = re.sub(r"(?<=[a-z0-9])['\-](?=[a-z0-9])", "", text)
    out = re.sub(r"[^a-z0-9]+", " ", text).split()
    while out and out[0] in ARTICLES:
        out = out[1:]
    return out


def singulars(word):
    """The word and what it may be the plural of ("cookies": cookie, cooky)."""
    out = {word}
    if word in IRREGULAR:
        out.add(IRREGULAR[word])
    if len(word) > 3 and word.endswith("ies"):
        out.add(word[:-3] + "y")
    if len(word) > 3 and word.endswith("ves"):
        out |= {word[:-3] + "f", word[:-3] + "fe"}
    if len(word) > 3 and word.endswith("es"):
        out.add(word[:-2])
    if len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
        out.add(word[:-1])
    return out


def forms(text):
    """Comparable forms of a name: sorted words, the last one maybe singular,
    with and without a container word ("can of cola", "tuna can")."""
    ws = words(text)
    if len(ws) > 2 and ws[1] == "of" and singulars(ws[0]) & CONTAINERS:
        ws = ws[2:]
    variants = [ws]
    if len(ws) > 1 and singulars(ws[-1]) & CONTAINERS:
        variants.append(ws[:-1])
    out = set()
    for variant in variants:
        if variant:
            for last in singulars(variant[-1]):
                out.add(tuple(sorted(variant[:-1] + [last])))
    return out


def is_person(text):
    return any(singulars(w) & PERSON_WORDS for w in words(text))


def singular(phrase):
    """Best guess of the singular of a noun phrase, for answers."""
    ws = words(phrase) or ["object"]
    last = ws[-1]
    if last in IRREGULAR:
        last = IRREGULAR[last]
    elif last.endswith("ies") and last not in {"cookies", "movies"}:
        last = last[:-3] + "y"
    elif re.search(r"(ch|sh|x|ss)es$", last):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith("ss"):
        last = last[:-1]
    return " ".join(ws[:-1] + [last])


def plural(phrase):
    ws = words(singular(phrase))
    last = ws[-1]
    plurals = {v: k for k, v in IRREGULAR.items()}
    if last in plurals:
        last = plurals[last]
    elif re.search(r"[^aeiou]y$", last):
        last = last[:-1] + "ies"
    elif re.search(r"(ch|sh|x|s|z)$", last):
        last += "es"
    else:
        last += "s"
    return " ".join(ws[:-1] + [last])


def where(spec, location):
    """ "in the kitchen" for a room, "on the desk" for a piece of furniture."""
    preposition = "in" if location in spec.rooms else "on"
    return f"{preposition} the {location.replace('_', ' ')}"


def count_answer(text, labels, place):
    """ "There are 3 drinks on the desk: green tea, lemonade and red bull." """
    n = len(labels)
    if n == 0:
        return f"There are no {plural(text)} {place}."
    if n == 1:
        return f"There is one {singular(text)} {place}: the {labels[0]}."
    names = f": {', '.join(labels[:-1])} and {labels[-1]}" if n <= 5 else ""
    return f"There are {n} {plural(text)} {place}{names}."


# -- what a command asks for -------------------------------------------------


@dataclass(frozen=True)
class Target:
    """Objects a name of a command stands for.

    kind: "any" (object, item, ...), "objects" (labels and categories), or
    "unknown" (not in the object list).
    """

    kind: str
    labels: frozenset = frozenset()
    categories: frozenset = frozenset()

    def matches(self, label, category):
        if self.kind == "any":
            return True
        if norm_label(label) in {norm_label(v) for v in self.labels}:
            return True
        return norm_label(category) in {norm_label(c) for c in self.categories}


def resolve_target(text, object_list):
    """Target of a name: generic word, label (or an alias), category, group."""
    found = forms(text)
    if not found or found & {(w,) for w in GENERIC}:
        return Target("any")
    labels = {
        o.label
        for o in object_list.objects
        if any(forms(name) & found for name in (o.label, *o.aliases))
    }
    if labels:
        return Target("objects", frozenset(labels))
    categories = {o.category for o in object_list.objects if forms(o.category) & found}
    if categories:
        return Target("objects", categories=frozenset(categories))
    known = {o.category for o in object_list.objects}
    for name, members in object_list.groups:
        if forms(name) & found:
            return Target(
                "objects",
                frozenset(m for m in members if m not in known),
                frozenset(m for m in members if m in known),
            )
    return Target("unknown")


def parse_superlative(text):
    """(superlative, subject) of "the heaviest drink", or ValueError."""
    ws = words(text)
    for i, w in enumerate(ws):
        if w in PREPOSITIONS and i > 0:
            ws = ws[:i]  # "smallest item on the side tables"
            break
    found, rest, i = [], [], 0
    while i < len(ws):
        for n in (3, 2, 1):  # the longest phrase first: "weighs the most"
            phrase = tuple(ws[i : i + n])
            if phrase in PHRASES:
                found.append(PHRASES[phrase])
                i += len(phrase)
                break
        else:
            rest.append(ws[i])
            i += 1
    if len({SUPERLATIVES[f] for f in found}) != 1:
        raise ValueError(
            f"needs one of {', '.join(SUPERLATIVES)} (or a synonym): {text!r}"
        )
    rest = [w for w in rest if w not in FILLER]
    if rest and rest[0] == "of":
        rest = rest[1:]  # "the heaviest of the drinks"
    subject = " ".join(rest) or "object"
    return found[0], subject


def measure(name, percept, entry):
    """Value of a percept for a superlative; None if unknown (no weight).

    The size is the object list's for a listed label (`entry`, the object
    list's ObjectType): a camera detector's box covers only the visible part.
    Other labels use the detected box.
    """
    kind = SUPERLATIVES[name][0]
    size = entry.size if entry is not None else percept.size
    if kind == "volume":
        return size[0] * size[1] * size[2]
    if kind == "weight":
        return entry.weight if entry is not None else None
    return tuple(sorted(size)[:2])  # thickness, then the next smallest side


def rank(candidates, name, entry_of):
    """Candidates [(percept, ...)] best first for the superlative, then by
    detections and confidence; those without a value are left out.
    `entry_of(label)` gives the label's object list entry or None."""
    valued = [(measure(name, c[0], entry_of(c[0].label)), c) for c in candidates]
    valued = [(value, c) for value, c in valued if value is not None]
    # Stable sorts: the value decides, ties go to more detections, then confidence.
    valued.sort(key=lambda v: (v[1][0].count, v[1][0].confidence), reverse=True)
    valued.sort(key=lambda v: v[0], reverse=SUPERLATIVES[name][1])
    return [c for _, c in valued]


# -- places ------------------------------------------------------------------


def places_of(spec, location):
    """[(furniture location, furniture name or None)] that `location` covers:
    the furniture of a room, or the furniture location itself."""
    if location in spec.rooms:
        return list(spec.rooms[location].items())
    for places in spec.rooms.values():
        if location in places:
            return [(location, places[location])]
    return []


def visit_order(start, points):
    """Labels of `points` ({label: (x, y)}) in the order of the shortest
    straight-line path from `start`; nearest first beyond 7 points."""
    labels = list(points)
    if len(labels) > 7:
        order, here = [], start
        while labels:
            nearest = min(labels, key=lambda k: math.dist(here, points[k]))
            labels.remove(nearest)
            order.append(nearest)
            here = points[nearest]
        return order

    def length(order):
        stops = [start] + [points[k] for k in order]
        return sum(math.dist(a, b) for a, b in zip(stops, stops[1:]))

    return list(min(itertools.permutations(labels), key=length)) if labels else []


def placed_record(percept, place, frame, level):
    """The object record of aha_tasks.objects with its furniture location
    ("surface") and level."""
    return {**record(percept, frame), "surface": place, "level": level}


# -- skills ------------------------------------------------------------------


class _ObjectQuery(Skill):
    """Common parts: params, places, the look over them and speech."""

    TARGET = "target"  # param of the name of the objects
    OUTPUT = None  # default memory key of the output; None: `output` is required

    @classmethod
    def validate(cls, params, spec):
        required = {cls.TARGET, "location"}
        if cls.OUTPUT is None:
            required.add("output")
        check_keys(params, required, {"output", "margin"} - required)
        check_name(params, cls.TARGET)
        location = params["location"]
        if not isinstance(location, str) or location not in spec.locations:
            raise ValueError(f"Unknown location: {location!r}")
        if "output" in params:
            check_name(params, "output")
        if "margin" in params:
            check_margin(params)
        if is_person(params[cls.TARGET]):
            return  # unsupported(): a plan mocks the step, at any location
        if not places_of(spec, location):
            raise ValueError(
                f"{location} is not a room or a furniture location of the task's rooms"
            )
        cls.check_target(params[cls.TARGET])

    @classmethod
    def check_target(cls, text):
        """Raise ValueError for a name of objects this skill cannot do."""

    def ready(self):
        """Load the configs; every furniture of `rooms` must be in the furniture
        config."""
        objects = Objects.of(self.runtime)
        for room, places in self.runtime.spec.rooms.items():
            for place, furniture in places.items():
                if furniture is not None and objects.furniture.get(furniture) is None:
                    raise ValueError(
                        f"rooms.{room}.{place}: {furniture} is not in the furniture "
                        f"config ({objects.furniture.names})"
                    )

    def unsupported(self, params):
        if is_person(params.get(self.TARGET, "")):
            return "people are found and told about by the person skills"
        places = places_of(self.runtime.spec, params["location"])
        if not any(furniture for _, furniture in places):
            names = ", ".join(place for place, _ in places)
            return f"no furniture top to look at for {names}"
        return None

    def expected(self, params):
        return {LOCATION: params["location"]}

    def entry(self, label):
        """The object list's entry of a detected label, or None."""
        object_list = Objects.of(self.runtime).object_list
        return object_list.get(label) or next(
            (
                o
                for o in object_list.objects
                if norm_label(o.label) == norm_label(label)
            ),
            None,
        )

    def say(self, ctx, text):
        ok, message = Interaction.of(self.runtime).say(text, ctx.cancelled)
        if not ok:
            ctx.log(f"Could not say {text!r}: {message}")

    def ordered_places(self, ctx, location):
        """Furniture locations with a top, on the shortest path from the robot."""
        spec = self.runtime.spec
        places = places_of(spec, location)
        skipped = [place for place, furniture in places if furniture is None]
        if skipped:
            ctx.log(f"Not looked at (no furniture top): {', '.join(skipped)}")
        tops = {
            place: furniture for place, furniture in places if furniture is not None
        }
        try:
            start = resolve_pose(spec, ctx.memory.get(LOCATION))[:2]
        except ValueError:  # no or an unknown robot.location
            start = (0.0, 0.0)
        points = {place: resolve_pose(spec, place)[:2] for place in tops}
        return [(place, tops[place]) for place in visit_order(start, points)]

    def look_over(self, ctx, text, location, first=False):
        """Go to each furniture location of `location`, scan its top and keep
        the objects that match the name `text`.

        Returns ([(percept, place, frame, level)], places looked at,
        {robot.location: the place where the robot stands} or {} if it is at
        none of them), or a failed SkillResult. `first`: stop after the first
        place with a match.
        """
        objects = Objects.of(self.runtime)
        target = resolve_target(text, objects.object_list)
        if target.kind == "unknown":
            return SkillResult(False, f"{text} is not in the object list")
        margin = float(ctx.params.get("margin", MARGIN))
        here = ctx.memory.get(LOCATION)
        found, places = [], []
        for place, furniture in self.ordered_places(ctx, location):
            if place != here and self.runtime.navigation is not None:
                moved = go_to(self.runtime, place, ctx.cancelled)
                if not moved.success:
                    return SkillResult(
                        False, f"Could not go to {place}: {moved.message}"
                    )
                here = place
            try:
                frame, region, on, _ = objects.scan(
                    furniture, margin, ctx.cancelled, ctx.log
                )
            except Cancelled:
                return SkillResult(False, "Cancelled")
            except (RuntimeError, ValueError) as error:
                return SkillResult(False, f"{place}: {error}")
            ctx.log(
                f"On {place} ({region.name}): "
                + (", ".join(p.label for p, _ in on) or "nothing")
            )
            places.append(place)
            found += [
                (p, place, frame, level)
                for p, level in on
                if target.matches(p.label, p.category)
            ]
            if first and found:
                break
        at = {LOCATION: here} if here in places else {}
        return found, places, at


class FindObject(_ObjectQuery):
    """Find an object by name or category in a room or on a piece of furniture.

    params: {"object": "<name or category>", "location": "<room or furniture
    location>", "output": "object", "margin": 0.1}. Visits the furniture of a
    room until an object matches; outputs `<output>` (placed_record) and
    `robot.location`. Says what it found, and looks at it.
    """

    TARGET = "object"
    OUTPUT = "object"

    def execute(self, ctx):
        params = ctx.params
        text, location = params["object"].strip(), params["location"]
        looked = self.look_over(ctx, text, location, first=True)
        if isinstance(looked, SkillResult):
            return looked
        found, places, at = looked
        spec = self.runtime.spec
        if not found:
            self.say(ctx, f"I could not find any {text} {where(spec, location)}.")
            return SkillResult(False, f"No {text} on {', '.join(places)}")
        best = max(found, key=lambda item: (item[0].count, item[0].confidence))
        percept, place, frame, _ = best
        Objects.of(self.runtime).look_at(frame, percept.position)
        self.say(ctx, f"I found the {percept.label} {where(spec, place)}.")
        outputs = {params.get("output", self.OUTPUT): placed_record(*best), **at}
        return SkillResult(True, f"{percept.label} {where(spec, place)}", outputs)


class CountObjects(_ObjectQuery):
    """Count the objects of a category (or name) in a room or on furniture.

    params: {"target": "drinks", "location": "desk", "output": "cmd2.count",
    "margin": 0.1}. Outputs `<output>` (the number), `<output>.labels` and
    `<output>.answer` ("There are 3 drinks on the desk: ..."), and says it.
    """

    def execute(self, ctx):
        params = ctx.params
        text, location = params["target"].strip(), params["location"]
        output = params["output"]
        looked = self.look_over(ctx, text, location)
        if isinstance(looked, SkillResult):
            return looked
        found, _, at = looked
        labels = sorted(p.label for p, *_ in found)
        answer = count_answer(text, labels, where(self.runtime.spec, location))
        self.say(ctx, answer)
        outputs = {
            output: len(labels),
            f"{output}.labels": labels,
            f"{output}.answer": answer,
            **at,
        }
        return SkillResult(True, answer, outputs)


class DescribeObject(_ObjectQuery):
    """Tell the biggest / largest / smallest / heaviest / lightest / thinnest
    object (or one of a category) in a room or on furniture.

    params: {"target": "heaviest drink", "location": "desk", "output":
    "cmd1.answer", "margin": 0.1}. Outputs `<output>` (the label),
    `<output>.object` (placed_record), `<output>.answer` ("The heaviest drink
    on the desk is the green tea.") and says it.
    """

    @classmethod
    def check_target(cls, text):
        parse_superlative(text)

    def execute(self, ctx):
        params = ctx.params
        location, output = params["location"], params["output"]
        superlative, subject = parse_superlative(params["target"])
        looked = self.look_over(ctx, subject, location)
        if isinstance(looked, SkillResult):
            return looked
        found, _, at = looked
        spec = self.runtime.spec
        if not found:
            return SkillResult(False, f"No {subject} {where(spec, location)}")
        ranked = rank(found, superlative, self.entry)
        if not ranked:
            return SkillResult(False, f"No weight of {subject} in the object list")
        ctx.log(
            f"{superlative}: "
            + ", ".join(
                f"{p.label} {measure(superlative, p, self.entry(p.label))}"
                for p, *_ in ranked
            )
        )
        percept, _, frame, _ = ranked[0]
        Objects.of(self.runtime).look_at(frame, percept.position)
        answer = (
            f"The {superlative} {singular(subject)} {where(spec, location)} "
            f"is the {percept.label}."
        )
        self.say(ctx, answer)
        outputs = {
            output: percept.label,
            f"{output}.object": placed_record(*ranked[0]),
            f"{output}.answer": answer,
            **at,
        }
        return SkillResult(True, answer, outputs)
