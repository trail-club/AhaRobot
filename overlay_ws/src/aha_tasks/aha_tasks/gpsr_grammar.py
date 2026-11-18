"""Parse GPSR commands in the official generator's phrasing into command steps.

The command generator (RoboCupAtHome/CommandGenerator, gpsr_commands.py)
builds a command from one of 22 sentence patterns and chains follow-ups
("... and take it and put it on the desk"). This parser recognizes exactly
those patterns with the arena's words (aha_tasks.gpsr_vocabulary) and returns
the steps of the command in the form the LLM planner produces
(aha_tasks.gpsr_planner): [{"action": ..., "params": {...}}, ...], without
memory keys. Anything else raises ParseError; rephrased commands are left to
the LLM.
"""

import re

from aha_tasks.gpsr_vocabulary import (
    COMPARISONS,
    GARMENTS,
    GESTURE_PERSONS,
    GESTURE_PERSONS_PLURAL,
    OPERATOR,
    PERSON_INFO,
    POSE_PERSONS,
    POSE_PERSONS_PLURAL,
    TALK,
    COLOURS,
    article,
    spoken,
)

VERBS = {
    "go": ("go", "navigate"),
    "take": ("take", "get", "grasp", "fetch"),
    "place": ("put", "place"),
    "deliver": ("bring", "give", "deliver"),
    "bring": ("bring", "give"),
    "find": ("find", "locate", "look for"),
    "talk": ("tell", "say"),
    "greet": ("greet", "salute", "say hello to", "introduce yourself to"),
    "guide": ("guide", "escort", "take", "lead"),
}
GREETING = "Hello, I am AhaRobot."
NAMED_GREETING = "Hello {name}, I am AhaRobot."


class ParseError(ValueError):
    pass


def normalize(text):
    """Lower case words without punctuation: "T-shirt, please." -> "t shirt please"."""
    words = str(text).lower().replace("t-shirt", "t shirt").replace("tshirt", "t shirt")
    words = words.replace("'", "").replace("’", "")
    words = re.sub(r"[^a-z0-9 ]+", " ", words)
    words = " ".join(words.split())
    # "The first command is ..." as an operator may start.
    return re.sub(r"^(?:the )?(?:first|second|third) command (?:is )?", "", words)


def _alt(words):
    ordered = sorted({w.lower() for w in words}, key=len, reverse=True)
    return "(?:" + "|".join(re.escape(w) for w in ordered) + ")"


def _navigate(label):
    return {"action": "navigate", "params": {"location": label}}


def _person(phrase):
    """find_described_person params of a person phrase."""
    if phrase in GESTURE_PERSONS or phrase in GESTURE_PERSONS_PLURAL:
        return {"gesture": phrase}
    if phrase in POSE_PERSONS or phrase in POSE_PERSONS_PLURAL:
        return {"pose": phrase}
    return {"clothing": phrase}


class Parser:
    """Recognizes the generator's patterns over one vocabulary."""

    def __init__(self, vocabulary):
        self.vocabulary = v = vocabulary
        self.names = {name.lower(): name for name in v.names}
        rooms = [spoken(r) for r in v.rooms]
        placements = [spoken(p) for p in v.placements]
        locations = placements + [spoken(b) for b in v.beacons]
        self.words = {
            "ROOM": _alt(rooms),
            "LOC": _alt(locations),
            "PLCMT": _alt(placements),
            "LOCROOM": _alt(locations + rooms),
            "NAME": _alt(v.names),
            "OBJ": _alt(v.objects),
            "OBJCAT": _alt(list(v.objects) + list(v.singular_categories)),
            "SINGCAT": _alt(v.singular_categories),
            "PLURCAT": _alt(v.plural_categories),
            "PERS": _alt(GESTURE_PERSONS + POSE_PERSONS),
            "PERSPLUR": _alt(GESTURE_PERSONS_PLURAL + POSE_PERSONS_PLURAL),
            "GEST": _alt(GESTURE_PERSONS),
            "CLOTHE": _alt(f"{c} {g[0]}" for c in COLOURS for g in GARMENTS),
            "CLOTHES": _alt(f"{c} {g[1]}" for c in COLOURS for g in GARMENTS),
            "INFO": _alt(PERSON_INFO),
            "COMP": _alt(COMPARISONS),
            "TALK": _alt(TALK),
            "ART": "(?:a|an)",
            **{f"V{k.upper()}": _alt(words) for k, words in VERBS.items()},
        }
        pattern = self._compile
        self.main = [
            (pattern("{VGO} to the (?P<loc>{LOCROOM}) then (?P<rest>.+)"), self.go_to),
            (
                pattern(
                    "{VTAKE} {ART} (?P<obj>{OBJCAT}) from the (?P<plc>{PLCMT}) "
                    "and (?P<rest>.+)"
                ),
                self.take_from,
            ),
            (
                pattern(
                    "{VFIND} a (?P<prs>{PERS}) in the (?P<room>{ROOM}) and (?P<rest>.+)"
                ),
                self.find_person_in,
            ),
            (
                pattern(
                    "{VFIND} {ART} (?P<obj>{OBJCAT}) in the (?P<room>{ROOM}) "
                    "then (?P<rest>.+)"
                ),
                self.find_object_in,
            ),
            (
                pattern(
                    "meet (?P<name>{NAME}) in the (?P<room>{ROOM}) and (?P<rest>.+)"
                ),
                self.meet_in,
            ),
            (
                pattern(
                    "tell me how many (?P<cat>{PLURCAT}) there are on the (?P<plc>{PLCMT})"
                ),
                self.count_objects,
            ),
            (
                pattern(
                    "tell me how many (?P<prs>{PERSPLUR}) are in the (?P<room>{ROOM})"
                ),
                self.count_people,
            ),
            (
                pattern(
                    "tell me the (?P<info>{INFO}) of the person "
                    "(?:in the (?P<room>{ROOM})|at the (?P<loc>{LOC}))"
                ),
                self.tell_person_info,
            ),
            (
                pattern(
                    "tell me what is the (?P<comp>{COMP}) object on the (?P<plc>{PLCMT})"
                ),
                self.tell_object_property,
            ),
            (
                pattern(
                    "{VTALK} (?P<talk>{TALK}) to the (?P<prs>{GEST}) in the (?P<room>{ROOM})"
                ),
                self.talk_to_person,
            ),
            (
                pattern(
                    "follow (?P<name>{NAME}) from the (?P<loc>{LOC}) "
                    "to the (?P<dest>{ROOM})"
                ),
                self.follow_name,
            ),
            (
                pattern(
                    "{VGUIDE} (?P<name>{NAME}) from the (?P<loc>{LOC}) "
                    "to the (?P<dest>{LOCROOM})"
                ),
                self.guide_name,
            ),
            (
                pattern(
                    "{VGUIDE} the (?P<prs>{PERS}) from the (?P<loc>{LOC}) "
                    "to the (?P<dest>{LOCROOM})"
                ),
                self.guide_person,
            ),
            (
                pattern(
                    "{VGUIDE} the person wearing {ART} (?P<cloth>{CLOTHE}) from the "
                    "(?P<loc>{LOC}) to the (?P<dest>{LOCROOM})"
                ),
                self.guide_clothed,
            ),
            (
                pattern("{VBRING} me {ART} (?P<obj>{OBJ}) from the (?P<plc>{PLCMT})"),
                self.bring_me,
            ),
            (
                pattern(
                    "tell me what is the (?P<comp>{COMP}) (?P<cat>{SINGCAT}) "
                    "on the (?P<plc>{PLCMT})"
                ),
                self.tell_category_property,
            ),
            (
                pattern(
                    "{VGREET} the person wearing {ART} (?P<cloth>{CLOTHE}) "
                    "in the (?P<room>{ROOM}) and (?P<rest>.+)"
                ),
                self.greet_clothed,
            ),
            (
                pattern(
                    "{VGREET} (?P<name>{NAME}) in the (?P<room>{ROOM}) and (?P<rest>.+)"
                ),
                self.greet_name,
            ),
            (
                pattern(
                    "meet (?P<name>{NAME}) at the (?P<loc>{LOC}) then {VFIND} them "
                    "in the (?P<room>{ROOM})"
                ),
                self.meet_then_find,
            ),
            (
                pattern(
                    "tell me how many people in the (?P<room>{ROOM}) are wearing "
                    "(?P<cloth>{CLOTHES})"
                ),
                self.count_clothed,
            ),
            (
                pattern(
                    "tell the (?P<info>{INFO}) of the person at the (?P<loc>{LOC}) "
                    "to the person at the (?P<loc2>{LOC})"
                ),
                self.tell_info_to_person,
            ),
            (
                pattern(
                    "follow the (?P<prs>{PERS}) "
                    "(?:in the (?P<room>{ROOM})|at the (?P<loc>{LOC}))"
                ),
                self.follow_person_at,
            ),
        ]
        self.at_location = [
            (
                pattern("{VFIND} {ART} (?P<obj>{OBJCAT}) and (?P<rest>.+)"),
                self.find_object,
            ),
            (pattern("{VFIND} the (?P<prs>{PERS}) and (?P<rest>.+)"), self.find_person),
            (pattern("meet (?P<name>{NAME}) and (?P<rest>.+)"), self.meet),
        ]
        self.found_object = [(pattern("{VTAKE} it and (?P<rest>.+)"), self.take_it)]
        self.has_object = [
            (pattern("{VPLACE} it on the (?P<plc>{PLCMT})"), self.place_it),
            (pattern("throw it in the trash"), self.trash_it),
            (pattern("{VDELIVER} it to me"), self.deliver_to_me),
            (
                pattern("{VDELIVER} it to the (?P<prs>{PERS}) in the (?P<room>{ROOM})"),
                self.deliver_to_person,
            ),
            (
                pattern("{VDELIVER} it to (?P<name>{NAME}) in the (?P<room>{ROOM})"),
                self.deliver_to_name,
            ),
        ]
        self.found_person = [
            (pattern("{VTALK} (?P<talk>{TALK})"), self.talk),
            (pattern("follow them"), self.follow_them),
            (pattern("follow them to the (?P<dest>{LOCROOM})"), self.follow_them_to),
            (pattern("{VGUIDE} them to the (?P<dest>{LOCROOM})"), self.guide_them_to),
        ]

    def _compile(self, template):
        return re.compile("^" + template.format(**self.words) + "$")

    def parse(self, text):
        """Command steps of one command; ParseError if it is not canonical."""
        return self._match(self.main, normalize(text), {})

    def _match(self, patterns, text, context):
        for pattern, handler in patterns:
            match = pattern.match(text)
            if match:
                groups = {k: v for k, v in match.groupdict().items() if v is not None}
                if "name" in groups:  # as the vocabulary writes it
                    groups["name"] = self.names[groups["name"]]
                return handler(groups, context)
        raise ParseError(f"no generator pattern matches {text!r}")

    # --- Steps ---------------------------------------------------------------

    def _label(self, words):
        label = self.vocabulary.label(words)
        if label is None:
            raise ParseError(f"unknown location {words!r}")
        return label

    def _go(self, words):
        return _navigate(self._label(words))

    def _find(self, location, **who):
        return {
            "action": "find_described_person",
            "params": {**who, "location": self._label(location)},
        }

    def _then(self, patterns, groups, context, steps):
        return steps + self._match(patterns, groups["rest"], context)

    def _report(self, action, params, location):
        """Find out something at a location and tell the operator."""
        return [
            self._go(location),
            {"action": action, "params": {**params, "location": self._label(location)}},
            _navigate(self.vocabulary.operator),
            {"action": "tell_info", "params": {"person": OPERATOR}},
        ]

    # --- Main patterns -------------------------------------------------------------

    def go_to(self, g, context):
        context = {**context, "location": g["loc"]}
        return self._then(self.at_location, g, context, [self._go(g["loc"])])

    def take_from(self, g, context):
        steps = [
            self._go(g["plc"]),
            {
                "action": "pick_object",
                "params": {"object": g["obj"], "surface": self._label(g["plc"])},
            },
        ]
        return self._then(self.has_object, g, {**context, "object": g["obj"]}, steps)

    def find_person_in(self, g, context):
        steps = [self._go(g["room"]), self._find(g["room"], **_person(g["prs"]))]
        return self._then(self.found_person, g, {**context, "person": g["prs"]}, steps)

    def find_object_in(self, g, context):
        context = {**context, "location": g["room"]}
        steps = [self._go(g["room"]), self._find_object(g["obj"], g["room"])]
        return self._then(self.found_object, g, {**context, "object": g["obj"]}, steps)

    def meet_in(self, g, context):
        steps = [self._go(g["room"]), self._find(g["room"], name=g["name"])]
        return self._then(self.found_person, g, {**context, "person": g["name"]}, steps)

    def count_objects(self, g, context):
        return self._report("count", {"target": g["cat"]}, g["plc"])

    def count_people(self, g, context):
        return self._report("count_people", _person(g["prs"]), g["room"])

    def tell_person_info(self, g, context):
        location = g.get("room") or g["loc"]
        return self._report("find_described_person", {"info": g["info"]}, location)

    def tell_object_property(self, g, context):
        target = f"{g['comp']} object"
        return self._report("describe", {"target": target}, g["plc"])

    def talk_to_person(self, g, context):
        return [
            self._go(g["room"]),
            self._find(g["room"], gesture=g["prs"]),
            {"action": "answer_question", "params": {"question": g["talk"]}},
        ]

    def follow_name(self, g, context):
        return [
            self._go(g["loc"]),
            self._find(g["loc"], name=g["name"]),
            {
                "action": "follow_person",
                "params": {"person": g["name"], "location": self._label(g["dest"])},
            },
        ]

    def _guide(self, who, person, g):
        return [
            self._go(g["loc"]),
            self._find(g["loc"], **who),
            {
                "action": "guide_person",
                "params": {"person": person, "location": self._label(g["dest"])},
            },
        ]

    def guide_name(self, g, context):
        return self._guide({"name": g["name"]}, g["name"], g)

    def guide_person(self, g, context):
        return self._guide(_person(g["prs"]), g["prs"], g)

    def guide_clothed(self, g, context):
        phrase = clothed(g["cloth"])
        return self._guide({"clothing": phrase}, phrase, g)

    def bring_me(self, g, context):
        return [
            self._go(g["plc"]),
            {
                "action": "pick_object",
                "params": {"object": g["obj"], "surface": self._label(g["plc"])},
            },
        ] + self.deliver_to_me(g, {"object": g["obj"]})

    def tell_category_property(self, g, context):
        target = f"{g['comp']} {g['cat']}"
        return self._report("describe", {"target": target}, g["plc"])

    def greet_clothed(self, g, context):
        phrase = clothed(g["cloth"])
        steps = [
            self._go(g["room"]),
            self._find(g["room"], clothing=phrase),
            {"action": "say", "params": {"text": GREETING}},
        ]
        return self._then(self.found_person, g, {**context, "person": phrase}, steps)

    def greet_name(self, g, context):
        name = g["name"]
        steps = [
            self._go(g["room"]),
            self._find(g["room"], name=name),
            {"action": "say", "params": {"text": NAMED_GREETING.format(name=name)}},
        ]
        return self._then(self.found_person, g, {**context, "person": name}, steps)

    def meet_then_find(self, g, context):
        return [
            self._go(g["loc"]),
            self._find(g["loc"], name=g["name"]),
            self._go(g["room"]),
            self._find(g["room"], name=g["name"]),
        ]

    def count_clothed(self, g, context):
        target = {"clothing": f"people wearing {g['cloth']}"}
        return self._report("count_people", target, g["room"])

    def tell_info_to_person(self, g, context):
        return [
            self._go(g["loc"]),
            self._find(g["loc"], info=g["info"]),
            self._go(g["loc2"]),
            {
                "action": "tell_info",
                "params": {
                    "person": f"person at the {g['loc2']}",
                    "location": self._label(g["loc2"]),
                },
            },
        ]

    def follow_person_at(self, g, context):
        location = g.get("room") or g["loc"]
        return [
            self._go(location),
            self._find(location, **_person(g["prs"])),
            {"action": "follow_person", "params": {"person": g["prs"]}},
        ]

    # --- Follow-ups ------------------------------------------------------------

    def _find_object(self, obj, location):
        return {
            "action": "find_object",
            "params": {"object": obj, "location": self._label(location)},
        }

    def find_object(self, g, context):
        steps = [self._find_object(g["obj"], context["location"])]
        return self._then(self.found_object, g, {**context, "object": g["obj"]}, steps)

    def find_person(self, g, context):
        steps = [self._find(context["location"], **_person(g["prs"]))]
        return self._then(self.found_person, g, {**context, "person": g["prs"]}, steps)

    def meet(self, g, context):
        steps = [self._find(context["location"], name=g["name"])]
        return self._then(self.found_person, g, {**context, "person": g["name"]}, steps)

    def take_it(self, g, context):
        steps = [{"action": "pick_object", "params": {"object": context["object"]}}]
        return self._then(self.has_object, g, context, steps)

    def _put(self, context, surface):
        return [
            _navigate(surface),
            {
                "action": "place_object",
                "params": {"object": context["object"], "surface": surface},
            },
        ]

    def place_it(self, g, context):
        return self._put(context, self._label(g["plc"]))

    def trash_it(self, g, context):
        return self._put(context, self.vocabulary.trash)

    def deliver_to_me(self, g, context):
        return [
            _navigate(self.vocabulary.operator),
            {
                "action": "deliver_object",
                "params": {"object": context["object"], "person": OPERATOR},
            },
        ]

    def _deliver_to(self, context, room, person, who):
        return [
            self._go(room),
            self._find(room, **who),
            {
                "action": "deliver_object",
                "params": {"object": context["object"], "person": person},
            },
        ]

    def deliver_to_person(self, g, context):
        return self._deliver_to(context, g["room"], g["prs"], _person(g["prs"]))

    def deliver_to_name(self, g, context):
        return self._deliver_to(context, g["room"], g["name"], {"name": g["name"]})

    def talk(self, g, context):
        return [{"action": "answer_question", "params": {"question": g["talk"]}}]

    def follow_them(self, g, context):
        return [{"action": "follow_person", "params": {"person": context["person"]}}]

    def follow_them_to(self, g, context):
        return [
            {
                "action": "follow_person",
                "params": {
                    "person": context["person"],
                    "location": self._label(g["dest"]),
                },
            }
        ]

    def guide_them_to(self, g, context):
        return [
            {
                "action": "guide_person",
                "params": {
                    "person": context["person"],
                    "location": self._label(g["dest"]),
                },
            }
        ]


def clothed(clothing):
    """'person wearing a red t shirt' of the generator's 'red t shirt'."""
    return f"person wearing {article(clothing)} {clothing}"
