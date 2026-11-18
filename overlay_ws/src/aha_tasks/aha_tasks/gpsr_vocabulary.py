"""Words of GPSR commands: the command generator's phrases and this arena's names.

The fixed phrases (gestures, postures, clothing, talk topics, ...) are those of
the official command generator (RoboCupAtHome/CommandGenerator). The arena's
words come from config/vocabulary/<task>.json: which task locations are rooms,
placement locations and other locations, and the object list of aha_perception
(config/objects/robocup_objects.yaml) with the spoken singular and plural of
its categories. People's names are the task's initial_state `gpsr.names`.
Location labels use "_" where the spoken word has a space ("kitchen_table" is
"kitchen table").
"""

from dataclasses import dataclass
import json
from pathlib import Path

GESTURE_PERSONS = (
    "waving person",
    "person raising their left arm",
    "person raising their right arm",
    "person pointing to the left",
    "person pointing to the right",
)
GESTURE_PERSONS_PLURAL = (
    "waving persons",
    "persons raising their left arm",
    "persons raising their right arm",
    "persons pointing to the left",
    "persons pointing to the right",
)
POSE_PERSONS = ("sitting person", "standing person", "lying person")
POSE_PERSONS_PLURAL = ("sitting persons", "standing persons", "lying persons")
PERSON_INFO = ("name", "pose", "gesture")
COMPARISONS = ("biggest", "largest", "smallest", "heaviest", "lightest", "thinnest")
TALK = (
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
COLOURS = ("blue", "yellow", "black", "white", "red", "orange", "gray")
GARMENTS = (
    ("t shirt", "t shirts"),
    ("shirt", "shirts"),
    ("blouse", "blouses"),
    ("sweater", "sweaters"),
    ("coat", "coats"),
    ("jacket", "jackets"),
)
OPERATOR = "operator"  # person param: the one who gave the commands
NAMES = "gpsr.names"  # initial_state key of people's names
VOCABULARY_KEYS = {
    "rooms",
    "placements",
    "beacons",
    "trash",
    "operator",
    "objects",
    "categories",
}


def spoken(label):
    """The words of a location label: kitchen_table -> kitchen table."""
    return label.replace("_", " ")


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def clothing_phrases(plural=False):
    """'person wearing a red t shirt', or 'people wearing red t shirts'."""
    if plural:
        return [f"people wearing {c} {g[1]}" for c in COLOURS for g in GARMENTS]
    return [
        f"person wearing {article(c)} {c} {g[0]}" for c in COLOURS for g in GARMENTS
    ]


@dataclass(frozen=True)
class Category:
    name: str  # as in the object list, e.g. "drink"
    singular: str  # spoken, e.g. "drink"
    plural: str  # spoken, e.g. "drinks"


@dataclass(frozen=True)
class Vocabulary:
    names: tuple
    rooms: tuple  # location labels
    placements: tuple  # location labels where objects can be placed
    beacons: tuple  # other location labels
    trash: str  # location label of "the trash"
    operator: str  # location label where the operator waits
    objects: tuple  # object labels (spoken words)
    categories: tuple  # Category
    object_categories: dict  # object label -> category name

    @property
    def locations(self):
        """Labels a command can name: rooms, placement and other locations."""
        return self.rooms + self.placements + self.beacons

    @property
    def singular_categories(self):
        return tuple(c.singular for c in self.categories)

    @property
    def plural_categories(self):
        return tuple(c.plural for c in self.categories)

    def label(self, words):
        """The location label of spoken words ('kitchen table'), or None."""
        label = "_".join(str(words).lower().split())
        return label if label in self.locations + (self.operator,) else None

    def hotwords(self):
        """Words to bias speech recognition to: names, places, objects."""
        words = list(self.names)
        words += [spoken(label) for label in self.locations]
        words += list(self.objects)
        words += [w for c in self.categories for w in (c.singular, c.plural)]
        return list(dict.fromkeys(words))


def _labels(data, key, path):
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{path}: {key} must be a list of labels")
    return tuple(value)


def share_path(package, relative):
    """A file installed in a package's share directory, else in the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        path = Path(get_package_share_directory(package)) / relative
        if path.exists():
            return path
    except (ImportError, LookupError, ValueError):
        pass
    # Source tree: overlay_ws/src/<package>/<relative> (tests on a host).
    return Path(__file__).resolve().parents[2] / package / relative


def vocabulary_path(name):
    """config/vocabulary/<name>.json of aha_tasks, or a JSON path."""
    if str(name).endswith(".json"):
        return Path(name)
    return share_path("aha_tasks", f"config/vocabulary/{name}.json")


def load_object_list(reference):
    """{label: category} from an object list YAML, '<package>/<path in share>'."""
    from aha_perception.object_list import load_object_list as load

    path = Path(reference)
    if not path.is_absolute():
        package, _, relative = str(reference).partition("/")
        path = share_path(package, relative)
    return {entry.label: entry.category for entry in load(path).objects}


def names_of(initial_state):
    """People's names of a task's initial_state (`gpsr.names`), possibly none."""
    names = initial_state.get(NAMES, [])
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"{NAMES} must be a list of names: {names!r}")
    return tuple(names)


def parse_vocabulary(data, objects, path="vocabulary", locations=None, names=()):
    """Vocabulary from the JSON data, {object label: category} and people's names.

    `locations` (the task's location labels), if given, must contain every
    label of the vocabulary.
    """
    if not isinstance(data, dict) or set(data) != VOCABULARY_KEYS:
        raise ValueError(f"{path}: needs exactly {sorted(VOCABULARY_KEYS)}")
    rooms = _labels(data, "rooms", path)
    placements = _labels(data, "placements", path)
    beacons = _labels(data, "beacons", path)
    for key in ("trash", "operator"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"{path}: {key} must be a location label")
    every = rooms + placements + beacons
    if len(set(every)) != len(every):
        raise ValueError(f"{path}: a location is listed twice")
    if data["trash"] not in every:
        raise ValueError(f"{path}: trash {data['trash']!r} is not a listed location")
    if locations is not None:
        unknown = sorted(set(every + (data["operator"],)) - set(locations))
        if unknown:
            raise ValueError(f"{path}: not locations of the task: {unknown}")
    forms = data["categories"]
    if not isinstance(forms, dict) or not all(
        isinstance(v, list) and len(v) == 2 and all(isinstance(w, str) for w in v)
        for v in forms.values()
    ):
        raise ValueError(f"{path}: categories maps a category to [singular, plural]")
    categories = []
    for name in dict.fromkeys(objects.values()):
        singular, plural = forms.get(name, (name, f"{name}s"))
        categories.append(Category(name, singular, plural))
    return Vocabulary(
        names=tuple(names),
        rooms=rooms,
        placements=placements,
        beacons=beacons,
        trash=data["trash"],
        operator=data["operator"],
        objects=tuple(objects),
        categories=tuple(categories),
        object_categories=dict(objects),
    )


def load_vocabulary(name, locations=None, names=()):
    """The vocabulary config/vocabulary/<name>.json (or a JSON path)."""
    path = vocabulary_path(name)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    objects = load_object_list(data.get("objects", ""))
    return parse_vocabulary(data, objects, str(path), locations, names)
