"""Appearance phrases of the VLM: agreeing views, choosing and telling features.

memorize_person asks the VLM several times and keeps what most of the views
asked for agree on (`consensus`); describe_person tells some of those features
in one sentence (`choose_features`, `sentence`). The rulebook (2027 §5.1) scores up to four
told attributes, +20 each when right and -20 when wrong, without listing the
kinds, so a wrong one costs as much as a right one earns: at most one phrase
per body part is told, the parts in PARTS order, hedged or negated phrases
are dropped, and so are leading verbs and articles ("wearing a") and the
details after "with". ROS-independent.
"""

from collections import Counter
import re

MAX_FEATURES = 4  # attributes the score sheet counts

# Body parts in the order they are told, with the head nouns that name them.
# Garments and hair cover large areas of the crop; small or rare items come
# later, shoes last because the crop often cuts them off.
PARTS = (
    (
        "upper",
        "t-shirt shirt polo blouse sweater jumper hoodie sweatshirt pullover "
        "jacket coat cardigan blazer vest top dress suit uniform",
    ),
    ("glasses", "glasses eyeglasses spectacles sunglasses"),
    ("headwear", "hat cap beanie helmet headscarf turban"),
    ("lower", "jeans pants trousers shorts skirt leggings chinos slacks"),
    ("hair", "hair ponytail bun braid braids bald"),
    ("facial_hair", "beard moustache mustache goatee"),
    ("bag", "bag backpack handbag purse"),
    ("neck", "scarf tie necklace"),
    ("shoes", "shoes sneakers boots sandals trainers"),
)
PART_OF = {noun: part for part, nouns in PARTS for noun in nouns.split()}
PART_ORDER = {part: i for i, (part, _) in enumerate(PARTS)}
# Head nouns that take no article ("blue jeans", "glasses").
PLURAL = set(
    "glasses eyeglasses spectacles sunglasses jeans pants trousers shorts "
    "leggings chinos slacks shoes sneakers boots sandals trainers braids".split()
)
SYNONYMS = {
    "tshirt": "t-shirt",
    "tee": "t-shirt",
    "tee-shirt": "t-shirt",
    "blonde": "blond",
    "gray": "grey",
}
# Words the VLM uses in turn for one kind of item (named by the first word):
# views vote on the kind and its colour, not on the word (vote_key).
KINDS = (
    "shirt t-shirt top polo blouse",
    "sweater jumper sweatshirt hoodie pullover",
    "jacket coat blazer cardigan",
    "pants trousers jeans chinos slacks",
    "hat cap beanie",
    "glasses eyeglasses spectacles",
    "moustache mustache",
    "shoes sneakers trainers",
)
KIND_OF = {noun: words.split()[0] for words in KINDS for noun in words.split()}
COLOURS = set(
    "black white grey red orange yellow green blue purple pink brown beige navy "
    "blond ginger silver gold maroon khaki teal turquoise cream tan olive".split()
)
SHADES = {"dark", "light", "pale", "bright"}
# "possible beard", "may have a beard", "looks like a hat": a hedged phrase
# would vote like a sure one and be said as the VLM put it.
HEDGES = re.compile(
    r"\b(possibl[ey]|maybe|perhaps|probabl[ey]|presumably|apparently|likely|"
    r"might|may|could|appears?|seems?|seemingly|looks?|unsure|uncertain|"
    r"unclear|partially|partly|hidden|unknown|or)\b"
)
NEGATIONS = re.compile(r"^(no|not|without)\b|\bno\b")
# Leading words that sentence() says itself ("is wearing", "has", articles):
# "wearing glasses", "has a beard", "a red t-shirt", "carrying a backpack".
LEAD = re.compile(
    r"^(?:(?:a|an|the|is|are|wears?|wearing|has|have|having|carr(?:y|ies|ying)|"
    r"hold(?:s|ing)?|sport(?:s|ing)|dressed|with|in|his|her|their|pair of)\s+)+"
)
# Held, not worn: told only of a bag ("carrying a backpack", "bag on shoulder");
# "carrying a black jacket" or "cap in hand" is not told at all.
HELD = re.compile(
    r"\b(?:carr(?:y|ies|ying|ied)|hold(?:s|ing)?|held)\b|"
    r"\b(?:in|on|over|under)\s+(?:(?:the|a|one|his|her|their|left|right)\s+)*"
    r"(?:hands?|arms?|shoulders?|laps?|wrists?)\b"
)
DETAIL = re.compile(r"\s+(with|on|in|over|under|and|featuring|showing)\s.*$")
MAX_WORDS = 4  # of the phrase without LEAD and DETAIL words


def normalize(phrase):
    """(part, phrase) of one VLM phrase, or None if it should not be told."""
    if not isinstance(phrase, str):
        return None
    words = " ".join(phrase.lower().replace("_", " ").split()).strip(" .,")
    if not words or HEDGES.search(words) or NEGATIONS.search(words):
        return None
    held = HELD.search(words)
    words = LEAD.sub("", words + " ").strip()  # "has a beard" -> "beard"
    words = DETAIL.sub("", words)  # "red t-shirt with a logo" -> "red t-shirt"
    # Two-word names first: "tee shirt" is not "t-shirt shirt".
    words = re.sub(r"\b(?:t|tee)[ -]shirt\b", "t-shirt", words)
    tokens = [SYNONYMS.get(t, t) for t in words.split()]
    if not tokens or len(tokens) > MAX_WORDS or tokens == ["hair"]:
        return None  # "hair" alone ("hair in a bun" without the bun) says nothing
    if tokens[0] == "bald":
        return "hair", "bald"
    part = PART_OF.get(tokens[-1])
    if part is None or (held and part != "bag"):
        return None
    return part, " ".join(tokens)


def vote_key(part, phrase):
    """What views must agree on for a normalized phrase: part, kind and colour.

    "red t-shirt", "red shirt" and "red top" share ("upper", "shirt", ("red",));
    a shade counts only without a colour ("dark grey pants" is grey, "dark
    hair" is dark).
    """
    *words, noun = phrase.split()
    colours = tuple(w for w in words if w in COLOURS)
    shades = tuple(w for w in words if w in SHADES)
    return part, KIND_OF.get(noun, noun), colours or shades


def plain(key):
    """The phrase of a vote_key alone ("red shirt"), or None if it says nothing."""
    part, kind, colours = key
    if part == "hair" and kind == "hair" and not colours:
        return None  # "hair"
    return " ".join(colours + (kind,))


def consensus(views, asked=0):
    """Phrases that more than half of the `asked` views agree on.

    `views` are lists of VLM phrases, None for a view that failed; a failed or
    missing view counts against every phrase (`asked` defaults to
    len(views)), so of 3 views asked 2 must agree and one alone keeps nothing.
    Phrases of a known body part vote per part on vote_key, so a garment that
    the views name differently still counts; the part keeps its most common
    key if a majority gave it, told as the most common phrase if a majority
    gave exactly that, else as colour and kind (`plain`). A feature that one
    view made up, or a part whose colour the views disagree on, is dropped.
    Other phrases need the same text. The result keeps the order in which the
    kept keys first came.
    """
    total = max(asked, len(views))
    by_group, by_key, first = {}, {}, {}
    for view in views:
        keys, texts = set(), set()
        for phrase in view or ():
            found = normalize(phrase)
            if found is None:
                text = " ".join(str(phrase).lower().split())
                group, key = ("", text), ("", text, ())
            else:
                text, group, key = found[1], found[0], vote_key(*found)
            if not text:
                continue
            if (group, key) not in keys:
                keys.add((group, key))
                by_group.setdefault(group, Counter())[key] += 1
                first.setdefault(key, len(first))
            if (key, text) not in texts:
                texts.add((key, text))
                by_key.setdefault(key, Counter())[text] += 1
    kept = []
    for keys in by_group.values():
        key, count = keys.most_common(1)[0]
        if 2 * count <= total:
            continue
        text, count = by_key[key].most_common(1)[0]
        text = text if 2 * count > total else plain(key)
        if text:
            kept.append((first[key], text))
    return [text for _, text in sorted(kept)]


def choose_features(appearance, limit=MAX_FEATURES):
    """At most `limit` (part, phrase), one per part, in PARTS order."""
    chosen = {}
    for phrase in appearance:
        found = normalize(phrase)
        if found is not None and found[0] not in chosen:
            chosen[found[0]] = found[1]
    ordered = sorted(chosen.items(), key=lambda item: PART_ORDER[item[0]])
    return ordered[:limit]


# Vowel letters said with a consonant sound take "a": "a uniform", "a used bag".
CONSONANT_SOUND = re.compile(r"^(?:uni|use|usu|uti|eu|ewe|one\b|once\b)")


def with_article(phrase):
    if phrase.split()[-1] in PLURAL:
        return phrase
    vowel = phrase[0] in "aeiou" and not CONSONANT_SOUND.match(phrase)
    return ("an " if vowel else "a ") + phrase


def join(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def sentence(subject, features, listener=""):
    """English sentence telling `listener` what `subject` looks like, or ""."""
    wearing, has, being = [], [], []
    for part, phrase in features:
        if part == "hair" and phrase.split()[-1] == "bald":
            being.append(phrase)  # "is mostly bald", not "has a mostly bald"
        elif part == "hair" and phrase.endswith("hair"):
            has.append(phrase)
        elif part in ("hair", "facial_hair", "bag"):
            has.append(with_article(phrase))
        else:
            wearing.append(with_article(phrase))
    clauses = []
    if wearing:
        clauses.append("is wearing " + join(wearing))
    if has:
        clauses.append("has " + join(has))
    if being:
        clauses.append("is " + join(being))
    if not clauses:
        return ""
    text = f"{subject} {join(clauses)}."
    if listener:
        return f"{listener}, {text}"  # "Emma, the other guest is ..."
    return text[0].upper() + text[1:]
