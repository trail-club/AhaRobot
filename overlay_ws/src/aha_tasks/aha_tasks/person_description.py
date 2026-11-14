"""People described the way GPSR commands do, matched against ClassifyPerson answers.

The phrases are those of the official command generator
(RoboCupAtHome/CommandGenerator, gpsr_commands.py): gestures ("waving person",
"person raising their left arm", "persons pointing to the right"), postures
("sitting person"), clothing ("person wearing a red t shirt", "people ...
wearing blue jackets") and names. Labels of ClassifyPerson ("raising_left_arm")
are accepted as well. Left and right are the person's own sides.
"""

from dataclasses import dataclass
import re

POSTURES = ("standing", "sitting", "lying")
GESTURES = (
    "waving",
    "raising_left_arm",
    "raising_right_arm",
    "pointing_left",
    "pointing_right",
)
GARMENT_TYPES = ("t shirt", "shirt", "blouse", "sweater", "coat", "jacket")
COLOURS = (
    "blue",
    "yellow",
    "black",
    "white",
    "red",
    "orange",
    "gray",
    "green",
    "purple",
    "pink",
    "brown",
    "beige",
)
# A gesture also matches these labels from ClassifyPerson.
ALSO_MATCHES = {
    "raising_left_arm": ("raising_both_arms",),
    "raising_right_arm": ("raising_both_arms",),
}

GESTURE_PHRASES = (
    (r"\bwaving\b", "waving"),
    (r"\braising (?:their |the |his |her )?left (?:arm|hand)\b", "raising_left_arm"),
    (r"\braising (?:their |the |his |her )?right (?:arm|hand)\b", "raising_right_arm"),
    (r"\bpointing (?:to |towards |at )?(?:the |their )?left\b", "pointing_left"),
    (r"\bpointing (?:to |towards |at )?(?:the |their )?right\b", "pointing_right"),
)
POSTURE_PHRASES = (
    (r"\bsitting\b|\bseated\b", "sitting"),
    (r"\bstanding\b", "standing"),
    (r"\blying\b|\blaying\b", "lying"),
)


def _words(text):
    return " ".join(re.sub(r"[_\-]", " ", str(text).lower()).split())


def garment_type(word):
    """GARMENT_TYPES entry for a garment word (plural, 't-shirt', 'tshirt'), or None."""
    for candidate in (word, word[:-1] if word.endswith("s") else None):
        if candidate in ("tshirt", "tee", "tee shirt"):
            candidate = "t shirt"
        if candidate in GARMENT_TYPES:
            return candidate
    return None


@dataclass(frozen=True)
class Description:
    gesture: str | None = None
    posture: str | None = None
    clothing: tuple[str, str] | None = None  # (colour, garment type)
    name: str | None = None

    @property
    def empty(self):
        return not (self.gesture or self.posture or self.clothing or self.name)

    @property
    def needs_vlm(self):
        """Clothing is only seen by the VLM."""
        return self.clothing is not None

    def text(self):
        parts = []
        if self.posture:
            parts.append(self.posture)
        if self.gesture:
            parts.append(self.gesture.replace("_", " "))
        if self.clothing:
            parts.append("wearing " + " ".join(self.clothing))
        if self.name:
            parts.append(f"named {self.name}")
        return " ".join(parts) or "anyone"

    def matches(self, name="", posture="", gesture="", garments=()):
        """True / False, or None when an answer the description needs is unknown."""
        results = []
        if self.name:
            results.append(name == self.name if name else None)
        if self.posture:
            results.append(posture == self.posture if posture else None)
        if self.gesture:
            accepted = (self.gesture, *ALSO_MATCHES.get(self.gesture, ()))
            results.append(gesture in accepted if gesture else None)
        if self.clothing:
            results.append(_wears(garments, *self.clothing) if garments else None)
        if False in results:
            return False
        if None in results:
            return None
        return True


def _wears(garments, colour, kind):
    for garment in garments:
        words = _words(garment).split()
        if (
            len(words) >= 2
            and words[0] == colour
            and garment_type(" ".join(words[1:])) == kind
        ):
            return True
    return False


def parse_phrase(text):
    """Description of one GPSR phrase, e.g. 'person raising their left arm'."""
    words = _words(text)
    if not words:
        raise ValueError("empty person description")
    label = words.replace(" ", "_")
    if label in GESTURES:
        return Description(gesture=label)
    if label in POSTURES:
        return Description(posture=label)
    gesture = next(
        (g for pattern, g in GESTURE_PHRASES if re.search(pattern, words)), None
    )
    posture = next(
        (p for pattern, p in POSTURE_PHRASES if re.search(pattern, words)), None
    )
    clothing = None
    match = re.search(r"\bwearing (?:an? |the )?(\w+) (.+)$", words)
    if match:
        colour = "gray" if match.group(1) == "grey" else match.group(1)
        kind = garment_type(match.group(2))
        if colour not in COLOURS or kind is None:
            raise ValueError(f"unknown clothing in {text!r}")
        clothing = (colour, kind)
    if gesture is None and posture is None and clothing is None:
        raise ValueError(f"no gesture, posture or clothing in {text!r}")
    return Description(gesture=gesture, posture=posture, clothing=clothing)


def parse(params):
    """Description from skill params: gesture / pose / clothing / description / name.

    Each value is a phrase; they combine, e.g. {"pose": "sitting person",
    "clothing": "person wearing a blue coat"}. Raises ValueError for unknown or
    contradicting phrases.
    """
    fields = {}
    for key in ("description", "gesture", "pose", "clothing"):
        if key not in params:
            continue
        if not isinstance(params[key], str):
            raise ValueError(f"{key} must be a phrase: {params[key]!r}")
        parsed = parse_phrase(params[key])
        for field in ("gesture", "posture", "clothing"):
            value = getattr(parsed, field)
            if value is None:
                continue
            if fields.get(field, value) != value:
                raise ValueError(f"contradicting {field}: {fields[field]!r}, {value!r}")
            fields[field] = value
    name = params.get("name")
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"name must be a nonempty string: {name!r}")
        fields["name"] = name.strip()
    return Description(**fields)
