"""Appearance phrases of the VLM: agreeing views, choosing and telling features.

memorize_person asks the VLM several times and keeps what most views agree on
(`consensus`); describe_person tells some of those features in one sentence
(`choose_features`, `sentence`). The rulebook (2027 §5.1) scores up to four
told attributes, +20 each when right and -20 when wrong, without listing the
kinds, so a wrong one costs as much as a right one earns: at most one phrase
per body part is told, the parts in PARTS order, and hedged or negated phrases
and the details after "with" are dropped. ROS-independent.
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
        "t-shirt shirt polo blouse sweater jumper hoodie sweatshirt jacket coat "
        "cardigan blazer vest top dress suit uniform",
    ),
    ("glasses", "glasses eyeglasses spectacles sunglasses"),
    ("headwear", "hat cap beanie helmet headscarf turban"),
    ("lower", "jeans pants trousers shorts skirt leggings chinos"),
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
    "leggings chinos shoes sneakers boots sandals trainers braids".split()
)
SYNONYMS = {"tshirt": "t-shirt", "tee": "t-shirt", "tee-shirt": "t-shirt"}
HEDGES = re.compile(
    r"\b(possibly|maybe|perhaps|probably|likely|might|appears?|seems?|"
    r"unclear|partially|partly|hidden|unknown|or)\b"
)
NEGATIONS = re.compile(r"^(no|not|without)\b|\bno\b")
DETAIL = re.compile(r"\s+(with|on|in|over|under|and|featuring|showing)\s.*$")
MAX_WORDS = 4


def normalize(phrase):
    """(part, phrase) of one VLM phrase, or None if it should not be told."""
    if not isinstance(phrase, str):
        return None
    words = " ".join(phrase.lower().replace("_", " ").split()).strip(" .,")
    if not words or HEDGES.search(words) or NEGATIONS.search(words):
        return None
    words = DETAIL.sub("", words)  # "red t-shirt with a logo" -> "red t-shirt"
    words = " ".join(SYNONYMS.get(t, t) for t in words.split())
    tokens = re.sub(r"\bt shirt\b", "t-shirt", words).split()
    if not tokens or len(tokens) > MAX_WORDS:
        return None
    if tokens[0] == "bald":
        return "hair", "bald"
    part = PART_OF.get(tokens[-1])
    return None if part is None else (part, " ".join(tokens))


def consensus(views):
    """Phrases of a strict majority of `views` (lists of VLM phrases).

    Phrases of a known body part vote per part: the part's most common phrase
    is kept if more than half of the views gave exactly it, so a feature
    that one view made up, or a colour the views disagree on, is dropped.
    Other phrases need the same text in more than half of the views. The
    result keeps the order in which the phrases first came.
    """
    views = [view for view in views if view is not None]
    by_part, first = {}, {}
    for view in views:
        seen = set()
        for phrase in view:
            found = normalize(phrase)
            key = found[0] if found else " ".join(str(phrase).lower().split())
            text = found[1] if found else key
            if not key or (key, text) in seen:
                continue
            seen.add((key, text))
            by_part.setdefault(key, Counter())[text] += 1
            first.setdefault(text, len(first))
    kept = []
    for key, phrases in by_part.items():
        text, count = phrases.most_common(1)[0]
        if 2 * count > len(views):
            kept.append(text)
    return sorted(kept, key=first.__getitem__)


def choose_features(appearance, limit=MAX_FEATURES):
    """At most `limit` (part, phrase), one per part, in PARTS order."""
    chosen = {}
    for phrase in appearance:
        found = normalize(phrase)
        if found is not None and found[0] not in chosen:
            chosen[found[0]] = found[1]
    ordered = sorted(chosen.items(), key=lambda item: PART_ORDER[item[0]])
    return ordered[:limit]


def with_article(phrase):
    if phrase.split()[-1] in PLURAL:
        return phrase
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def join(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def sentence(subject, features, listener=""):
    """English sentence telling `listener` what `subject` looks like, or ""."""
    wearing, has, being = [], [], []
    for part, phrase in features:
        if phrase == "bald":
            being.append("bald")
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
    text = text[0].upper() + text[1:]
    return f"{listener}, {text}" if listener else text
