"""What the robot says and hears in GPSR conversations; no ROS.

- Names: the RoboCup@Home names (CompetitionTemplate names/names.md) or the
  task's list in memory (`gpsr.names`). A transcript is matched to them
  exactly, then by spelling and sound (Soundex), since the English ASR model
  ignores hotwords ("Jewels" is Jules).
- Replies to "Are you Charlie?" and "What is your name?": yes / no and a name.
- Questions of the command generator's {talk} ("the time", "your teams name",
  ...) and their rephrasings, answered from the clock and the robot's facts
  in memory (`gpsr.robot_name`, `gpsr.team_*`, `gpsr.utc_offset`).
- Sentences for what the robot reports: a count of people, a person's name,
  pose or gesture, and any other value in memory.
- Dialogue: asks the person in front of the robot through an interaction
  object (aha_tasks.interaction.Interaction, or a fake): by voice, and for
  yes / no with buttons on the screen when the voice does not work.
"""

import datetime
import difflib
import re

from aha_tasks.interaction import NO_ASR

# RoboCup@Home CompetitionTemplate names/names.md, used by the command generator.
NAMES = (
    "Adel",
    "Angel",
    "Axel",
    "Charlie",
    "Jane",
    "Jules",
    "Morgan",
    "Paris",
    "Robin",
    "Simone",
)
NAMES_KEY = "gpsr.names"  # memory (initial_state): the names people may have
# Memory (initial_state) keys of the facts the robot tells about itself.
FACTS = {
    "robot_name": "gpsr.robot_name",
    "team_name": "gpsr.team_name",
    "team_country": "gpsr.team_country",
    "team_affiliation": "gpsr.team_affiliation",
}
UTC_OFFSET_KEY = "gpsr.utc_offset"  # hours; without it the system's time zone
PERSON_INFO = ("name", "pose", "gesture")

# -- names and replies --------------------------------------------------------

YES_WORDS = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "yea",
    "correct",
    "right",
    "sure",
    "indeed",
    "exactly",
    "affirmative",
    "certainly",
    "absolutely",
    "ok",
    "okay",
    "me",  # "That's me", "It's me"
}
NO_WORDS = {"no", "nope", "nah", "wrong", "incorrect", "negative", "never"}
# Asking back ("What?", "Pardon?", "Excuse me?"): no answer and no name.
QUERY_WORDS = {"what", "pardon", "huh", "excuse", "again", "repeat", "sorry"}
# Words that are not a name in a short reply.
FILLER = (
    YES_WORDS
    | NO_WORDS
    | QUERY_WORDS
    | {
        "not",
        "i",
        "i'm",
        "im",
        "am",
        "a",
        "the",
        "my",
        "name",
        "name's",
        "is",
        "it",
        "it's",
        "its",
        "this",
        "that",
        "that's",
        "call",
        "hello",
        "hi",
        "hey",
        "um",
        "uh",
        "oh",
        "well",
        "please",
        "thank",
        "thanks",
        "you",
        "sorry",
        "and",
    }
)
# A name follows these words.
NAME_CUE = re.compile(
    r"\b(?:my name is|my name's|name is|i am|i'm|im|call me|this is|it is|it's)\s+"
)
MIN_RATIO = 0.8  # difflib ratio for a misspelt name
# With the same Soundex code: "Jewels" is Jules (0.55), "John" is not Jane (0.5).
MIN_SOUND_RATIO = 0.54


def names_of(memory):
    """The names of the task (`gpsr.names`), else NAMES."""
    names = memory.get(NAMES_KEY) or NAMES
    if not isinstance(names, (list, tuple)) or not all(
        isinstance(n, str) and n.strip() for n in names
    ):
        raise ValueError(f"{NAMES_KEY} must be a list of names: {names!r}")
    return [n.strip() for n in names]


def _tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower().replace("’", "'"))


def soundex(word):
    """American Soundex code, e.g. "Jules" and "Jewels" are J420."""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    code, last = word[0].upper(), codes.get(word[0], "")
    for letter in word[1:]:
        digit = codes.get(letter, "")
        if digit and digit != last:
            code += digit
        if letter not in "hw":  # h and w do not separate equal codes
            last = digit
    return (code + "000")[:4]


def _closest(word, names):
    """(score, name) of the name that `word` is a misspelling of, or None."""
    best = None
    for name in names:
        lowered = name.lower()
        ratio = difflib.SequenceMatcher(None, word, lowered).ratio()
        if ratio >= MIN_RATIO or (
            ratio >= MIN_SOUND_RATIO and soundex(word) == soundex(lowered)
        ):
            if best is None or ratio > best[0]:
                best = (ratio, name)
    return best


def _cued(text):
    """The words right after a name cue ("my name is", "I'm", ...)."""
    lowered = " ".join(_tokens(text))
    return [_tokens(lowered[m.end() :])[:1] for m in NAME_CUE.finditer(lowered)]


def find_name(text, names):
    """The name of `names` said in `text`, or None.

    An exact name wins, the one after a cue ("I'm Jane, not Charlie") first.
    Otherwise a word after a cue, or a word of a short reply ("Jewels"), that
    is spelt or sounds like a name.
    """
    lowered = " ".join(_tokens(text))
    cued = {words[0] for words in _cued(text) if words}
    hits = []
    for name in names:
        match = re.search(rf"\b{re.escape(name.lower())}\b", lowered)
        if match:
            first = name.lower().split()[0]
            hits.append((first not in cued, match.start(), name))
    if hits:
        return min(hits)[2]
    words = sorted(cued)
    if not words:
        words = [t for t in _tokens(text) if t not in FILLER]
        if len(words) > 2:
            return None
    scored = [found for word in words if (found := _closest(word, names))]
    return max(scored)[1] if scored else None


def extract_name(text, names):
    """The name a person gave in `text`: from `names` if it sounds like one.

    A name not in `names` is taken only after a cue ("My name is Kenji"); a
    lone word that is no name ("What?", "Pardon?") is not a name.
    """
    name = find_name(text, names)
    if name is not None:
        return name
    cued = [words[0] for words in _cued(text) if words and words[0] not in FILLER]
    return cued[0].capitalize() if cued else None


# "That's me", "It's me", a lone "Me", "I am".
YES_PHRASE = re.compile(r"\b(?:that's|that is|it's|it is|this is|yes) me\b|^me$|^i am$")
# Denials that end the reply or name the person ("I'm not.", "Not me", "That
# isn't me"; "not <name>" is checked in parse_reply). A bare "not" or "n't" is
# no answer: "I didn't hear you", "I'm not sure", "Sure, why not".
NO_PHRASE = re.compile(
    r"^not(?: really| at all)?$|\b(?:i'm|im|i am|that's|that is|it's|it is) not$"
    r"|\b(?:isn't|ain't|wasn't) me\b|\bnot (?:me|him|her)\b"
)


def _answered_with(text, name):
    """Whether `name` is the answer itself: after a cue ("I'm Charlie") or alone.

    A name in a longer reply ("Charlie is over there", "This is Charlie's
    seat") is not.
    """
    if name.lower().split()[0] in {words[0] for words in _cued(text) if words}:
        return True
    rest = [t for t in _tokens(text) if t not in FILLER]
    return bool(rest) and all(_closest(t, [name]) for t in rest)


def parse_reply(text, names):
    """{"yes": bool, "no": bool, "name": <name of `names`> | None} of a reply.

    "name" is set only when the reply gives it as the answer (_answered_with).
    """
    tokens = _tokens(text)
    joined = " ".join(tokens)
    name = find_name(text, names)
    if name is not None and not _answered_with(text, name):
        name = None
    denied_name = name is not None and re.search(
        rf"\bnot {re.escape(name.lower())}\b", joined
    )
    # A yes word right after "not" ("I'm not sure") is no yes.
    negated = {i + 1 for i, t in enumerate(tokens) if t == "not" or t.endswith("n't")}
    return {
        "yes": any(
            t in YES_WORDS - {"me"} and i not in negated for i, t in enumerate(tokens)
        )
        or bool(YES_PHRASE.search(joined)),
        "no": any(t in NO_WORDS for t in tokens)
        or bool(NO_PHRASE.search(joined) or denied_name),
        "name": name,
    }


def is_named(reply, name):
    """Whether a reply to "Are you <name>?" says yes (True), no (False) or neither (None).

    A denial wins ("No, I'm Jane", "I'm not Charlie"), then a name ("Charlie",
    "I'm Jane"), then a yes ("Yes", "That's me").
    """
    if reply["no"]:
        return False
    if reply["name"] is not None:
        return reply["name"].lower() == name.lower()
    if reply["yes"]:
        return True
    return None


# -- questions ---------------------------------------------------------------

# In order: the first topic whose pattern is in the question.
TOPICS = (
    ("tomorrow", r"\btomorrow\b"),
    ("day_of_month", r"\bmonth\b"),
    ("day_of_week", r"\bweek\b|\bweekday\b"),
    ("time", r"\btime\b|\bclock\b|\bhours?\b"),
    (
        "team_affiliation",
        r"affiliat|universit|institut|organi[sz]ation|\bschool\b|\blab\b|\bbelong",
    ),
    ("team_country", r"\bcountry\b|\bnation|\bfrom\b"),
    ("team_name", r"\bteams?\b"),
    ("today", r"\btoday\b|\bdate\b|\bday\b"),
    (
        "about_yourself",
        r"\byourself\b|\babout you\b|\bintroduce\b|\bwho (?:are you|you are)\b"
        r"|\byour name\b",
    ),
)
TOPIC_FACTS = {
    "team_name": ("team_name",),
    "team_country": ("team_country",),
    "team_affiliation": ("team_affiliation",),
    "about_yourself": ("robot_name", "team_name", "team_affiliation", "team_country"),
}
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
# The generator's {talk}, for messages.
KNOWN_QUESTIONS = (
    "something about yourself",
    "the time",
    "what day is today",
    "what day is tomorrow",
    "your teams name",
    "your teams country",
    "your teams affiliation",
    "the day of the week",
    "the day of the month",
)


def question_topic(question):
    """The topic of a {talk} question or a rephrasing of it, or None."""
    text = str(question).lower().replace("’", "'").replace("_", " ")
    text = " ".join(re.sub(r"[^a-z0-9 ]", " ", text.replace("'s", "")).split())
    for topic, pattern in TOPICS:
        if re.search(pattern, text):
            return topic
    return None


def facts_of(memory):
    """{"robot_name": ..., "team_name": ...} from memory; missing ones are None."""
    return {fact: memory.get(key) for fact, key in FACTS.items()}


def missing_facts(topic, facts):
    """Memory keys the answer to `topic` needs but `facts` lacks."""
    return [
        FACTS[fact]
        for fact in TOPIC_FACTS.get(topic, ())
        if not isinstance(facts.get(fact), str) or not facts[fact].strip()
    ]


def local_now(memory):
    """The wall-clock time in `gpsr.utc_offset` hours, else the system's zone."""
    offset = memory.get(UTC_OFFSET_KEY)
    if offset is None:
        return datetime.datetime.now().astimezone()
    zone = datetime.timezone(datetime.timedelta(hours=offset))
    return datetime.datetime.now(zone)


def check_utc_offset(value):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not -12 <= value <= 14
    ):
        raise ValueError(f"{UTC_OFFSET_KEY} must be hours from UTC: {value!r}")


def ordinal(number):
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{'th' if 11 <= number % 100 <= 13 else suffix}"


def _date(day):
    return f"{WEEKDAYS[day.weekday()]}, {MONTHS[day.month - 1]} {ordinal(day.day)}, {day.year}"


def answer(topic, facts, now):
    """The sentence answering `topic` (question_topic); KeyError for an unknown one."""
    missing = missing_facts(topic, facts)
    if missing:
        raise ValueError(f"{', '.join(missing)} not set in initial_state")
    if topic == "time":
        hour = now.hour % 12 or 12
        return f"It is {hour}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}."
    if topic == "today":
        return f"Today is {_date(now)}."
    if topic == "tomorrow":
        return f"Tomorrow is {_date(now + datetime.timedelta(days=1))}."
    if topic == "day_of_week":
        return f"Today is {WEEKDAYS[now.weekday()]}."
    if topic == "day_of_month":
        return f"Today is the {ordinal(now.day)} of {MONTHS[now.month - 1]}."
    if topic == "team_name":
        return f"My team's name is {facts['team_name']}."
    if topic == "team_country":
        return f"My team is from {facts['team_country']}."
    if topic == "team_affiliation":
        return f"My team's affiliation is {facts['team_affiliation']}."
    if topic == "about_yourself":
        return (
            f"My name is {facts['robot_name']}. I am a service robot of team "
            f"{facts['team_name']} from {facts['team_affiliation']} in "
            f"{facts['team_country']}."
        )
    raise KeyError(topic)


# -- sentences to report -----------------------------------------------------

GESTURE_TEXT = {
    "waving": "waving",
    "raising_left_arm": "raising their left arm",
    "raising_right_arm": "raising their right arm",
    "raising_both_arms": "raising both arms",
    "pointing_left": "pointing to the left",
    "pointing_right": "pointing to the right",
}
POSTURE_TEXT = {"standing": "standing", "sitting": "sitting", "lying": "lying down"}
# Last words of location labels that are rooms ("in the" rather than "at the").
ROOM_WORDS = {
    "room",
    "bedroom",
    "bathroom",
    "kitchen",
    "office",
    "hall",
    "hallway",
    "corridor",
    "entrance",
    "garage",
    "study",
}


def person_info_sentence(info, value):
    """'The person's name is Charlie.' and the like; ValueError if unknown."""
    if info not in PERSON_INFO:
        raise ValueError(f"info must be one of {PERSON_INFO}: {info!r}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"the person's {info} is not known")
    if info == "name":
        return f"The person's name is {value.strip()}."
    if info == "pose":
        return f"The person is {POSTURE_TEXT.get(value, value.replace('_', ' '))}."
    if value == "none":
        return "The person is not making any gesture."
    return f"The person is {GESTURE_TEXT.get(value, value.replace('_', ' '))}."


def _number(value):
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _listing(items):
    items = [str(i).replace("_", " ") for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def value_sentence(key, value):
    """A sentence for the value of a memory key; ValueError if there is nothing to tell.

    Person information (`<role>.name` / `.pose` / `.gesture`) is phrased as
    such; other values as "The answer is ...".
    """
    suffix = key.rsplit(".", 1)[-1]
    if suffix in PERSON_INFO and isinstance(value, str):
        return person_info_sentence(suffix, value)
    if isinstance(value, bool):
        return "Yes." if value else "No."
    if isinstance(value, (int, float)):
        return f"The answer is {_number(value)}."
    if isinstance(value, dict):
        value = value.get("label") or value.get("name")
    if isinstance(value, str) and value.strip():
        return f"The answer is {value.strip().replace('_', ' ')}."
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return f"The answer is {_listing(value)}."
    raise ValueError(f"nothing to tell in {key}: {value!r}")


def place_phrase(location):
    """'in the living room', 'at the desk' for a location label."""
    words = location.replace("_", " ").split()
    return f"{'in' if words[-1] in ROOM_WORDS else 'at'} the {' '.join(words)}"


def _plural(noun):
    words = noun.split()
    last = words[-1]
    words[-1] = last + ("es" if re.search(r"(s|sh|ch|x)$", last) else "s")
    return " ".join(words)


def people_phrase(description, plural):
    """'sitting people wearing red t shirts', 'person raising their left arm'.

    `description` has gesture / posture / clothing as
    aha_tasks.person_description.Description.
    """
    before, after = [], []
    if description.posture:
        before.append(description.posture)
    if description.gesture == "waving":
        before.append("waving")
    elif description.gesture:
        after.append(GESTURE_TEXT.get(description.gesture, description.gesture))
    if description.clothing:
        colour, kind = description.clothing
        if plural:
            after.append(f"wearing {colour} {_plural(kind)}")
        else:
            article = "an" if colour[0] in "aeiou" else "a"
            after.append(f"wearing {article} {colour} {kind}")
    return " ".join([*before, "people" if plural else "person", *after])


def count_sentence(count, description, location=None):
    """'There are 2 waving people in the office.' for count_people."""
    where = f" {place_phrase(location)}" if location else ""
    if count == 0:
        return f"There are no {people_phrase(description, True)}{where}."
    if count == 1:
        return f"There is one {people_phrase(description, False)}{where}."
    return f"There are {count} {people_phrase(description, True)}{where}."


# -- dialogue ----------------------------------------------------------------

ATTEMPTS = 2  # questions by voice before giving up (or using the screen)
TOUCH_TIMEOUT = 10.0  # s for a touch of Yes / No on the screen


class Dialogue:
    """Ask the person in front of the robot.

    `interaction` has say(text, cancelled), listen(hotwords, cancelled,
    timeout) and ask(prompt, choices, cancelled, timeout) as
    aha_tasks.interaction.Interaction. Without an ASR, later questions skip the
    voice; `can_ask` turns false once neither the ASR nor the screen answers.
    """

    def __init__(
        self,
        interaction,
        names,
        cancelled,
        log,
        attempts=ATTEMPTS,
        listen_timeout=None,
        touch_timeout=TOUCH_TIMEOUT,
    ):
        self.interaction = interaction
        self.names = list(names)
        self.cancelled = cancelled
        self.log = log
        self.attempts = attempts
        self.listen_timeout = listen_timeout
        self.touch_timeout = touch_timeout
        self.voice = True  # cleared when the ASR is not served
        self.touch = touch_timeout > 0  # the screen may be used at all
        self.screen = self.touch  # cleared when the screen is not served

    @property
    def can_ask(self):
        return self.voice or self.screen

    def say(self, text):
        ok, message = self.interaction.say(text, self.cancelled)
        if not ok:
            self.log(f"Could not say {text!r}: {message}")

    def hear(self, hotwords):
        """The next utterance, or None (then the reason is logged)."""
        kwargs = {} if self.listen_timeout is None else {"timeout": self.listen_timeout}
        text, why = self.interaction.listen(hotwords, self.cancelled, **kwargs)
        if text is None:
            self.log(f"Nothing heard: {why}")
            if why.startswith(NO_ASR):
                self.voice = False
        return text

    def confirm(self, name):
        """Whether the person is `name`: True / False, or None if they could not answer."""
        hotwords = [*self.names, *([name] if name not in self.names else [])]
        question = f"Are you {name}?"
        if self.voice:
            self.say(f"Excuse me, are you {name}?")
        for attempt in range(self.attempts):
            if self.cancelled.is_set() or not self.voice:
                break
            text = self.hear(hotwords)
            if text is not None:
                decision = is_named(parse_reply(text, hotwords), name)
                verdict = {True: "yes", False: "no", None: "unclear"}[decision]
                self.log(f"Heard {text!r} to {question!r}: {verdict}")
                if decision is not None:
                    return self._thank(name, decision)
            if attempt + 1 < self.attempts and self.voice:
                self.say(
                    f"Sorry, I did not understand. {question} Please say yes or no."
                )
        if self.cancelled.is_set() or not self.screen:
            return None
        self.say(f"{question} Please touch Yes or No on my screen.")
        index, label = self.interaction.ask(
            question, ["Yes", "No"], self.cancelled, timeout=self.touch_timeout
        )
        if index is None:
            self.log(f"No touch: {label}")
            if label.startswith("No screen"):
                self.screen = False
            return None
        self.log(f"Touched {label!r} to {question!r}")
        return self._thank(name, index == 0)

    def _thank(self, name, decision):
        self.say(f"Nice to meet you, {name}." if decision else "Thank you.")
        return decision

    def ask_name(self):
        """The name the person gives (one of `names` if it sounds like one), or None."""
        if not self.voice:
            return None
        self.say("Excuse me, what is your name?")
        for attempt in range(self.attempts):
            if self.cancelled.is_set() or not self.voice:
                break
            text = self.hear(self.names)
            if text is not None:
                name = extract_name(text, self.names)
                self.log(f"Heard {text!r} as the name {name!r}")
                if name is not None:
                    self.say(f"Nice to meet you, {name}.")
                    return name
            if attempt + 1 < self.attempts and self.voice:
                self.say("Sorry, I did not catch your name. What is your name?")
        return None
