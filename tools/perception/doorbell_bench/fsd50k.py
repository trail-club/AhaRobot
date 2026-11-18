"""Select and decode the FSD50K clips used by the benchmark.

Reads the ground truth and metadata from Zenodo (record 4060432) and the audio
from the Opus mirror `philgzl/fsd50k` (see README), puts every clip into one
group and writes `clips/<split>/<fname>.npy` (STORE_RATE, float16) and
`clips.jsonl` under the data root.

Groups (first match wins):
  doorbell     label Doorbell. `category` comes from the title and tags
               (DOORBELL_RULES); `exclude` clips are not used at all.
  knock        label Knock.
  hard_chime   bells, chimes, alarms, telephones and other tonal signals.
  hard_knock   door, tap, wood, drum and other impacts that resemble a knock.
  ambient      speech, crowds, music and household sounds of a party.
Ambient clips are sampled to at most `--cap-ambient` per subgroup and split, hard
clips to `--cap-hard` per most specific matching label and split.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import (
    STORE_RATE,
    active_power,
    data_root,
    onset_seconds,
    resample,
    save_audio,
    write_jsonl,
)

# Ordered (category, pattern) rules over the lower-cased title and tags.
DOORBELL_RULES = [
    ("exclude", r"writing|escribiendo|bicycle|bike|fridge|keys,yard"),
    (
        "announcement",
        r"subway|metro|train|airport|terminal|elevator|\blift\b|announcement|stadium|"
        r"airplane|aircraft|seat ?belt|broadcast|signaali|\bpa\b|sliding.door|7'11",
    ),
    ("melody", r"melody|god save|beethoven|walk.renritt|samba|jingle|westminster"),
    (
        "handbell",
        r"\bpull\b|store bell|\bshop\b|hand-bell|servant|old.fashion|sonouvre",
    ),
    (
        "chime",
        r"ding|dong|chime|bing|bong|two-tone|tubular|electronic|intercom|entry phone",
    ),
    ("ring", r"buzz|ring|harsh|vintage|electric|mechanical|metallic|antique|old"),
]

HARD_CHIME = {
    "Bell",
    "Chime",
    "Church_bell",
    "Wind_chime",
    "Bicycle_bell",
    "Cowbell",
    "Glockenspiel",
    "Mallet_percussion",
    "Marimba_and_xylophone",
    "Tick-tock",
    "Clock",
    "Telephone",
    "Ringtone",
    "Alarm",
    "Microwave_oven",
    "Buzz",
    "Siren",
    "Piano",
    "Keyboard_(musical)",
}
HARD_KNOCK = {
    "Door",
    "Slam",
    "Sliding_door",
    "Cupboard_open_or_close",
    "Drawer_open_or_close",
    "Tap",
    "Wood",
    "Hammer",
    "Drum",
    "Snare_drum",
    "Bass_drum",
    "Clapping",
    "Tick",
}
AMBIENT = {
    "speech": {
        "Speech",
        "Conversation",
        "Chatter",
        "Crowd",
        "Laughter",
        "Applause",
        "Cheering",
        "Child_speech_and_kid_speaking",
        "Female_speech_and_woman_speaking",
        "Male_speech_and_man_speaking",
        "Singing",
    },
    "music": {"Music"},
    "household": {
        "Typing",
        "Dishes_and_pots_and_pans",
        "Cutlery_and_silverware",
        "Walk_and_footsteps",
        "Mechanical_fan",
        "Water_tap_and_faucet",
        "Toilet_flush",
        "Zipper_(clothing)",
        "Keys_jangling",
        "Packing_tape_and_duct_tape",
        "Chewing_and_mastication",
        "Domestic_sounds_and_home_sounds",
    },
}


def doorbell_category(title: str, tags: list[str]) -> str:
    text = (title + " " + ",".join(tags)).lower()
    for category, pattern in DOORBELL_RULES:
        if re.search(pattern, text):
            return category
    return "other"


def assign(labels: set[str], counts: Counter) -> tuple[str, str] | None:
    """Group and subgroup of a clip. FSD50K propagates labels to the ontology's
    parents, so the hard subgroup is the rarest (most specific) matching label."""
    if "Doorbell" in labels:
        return "doorbell", ""
    if "Knock" in labels:
        return "knock", ""
    for group, names in (("hard_chime", HARD_CHIME), ("hard_knock", HARD_KNOCK)):
        if labels & names:
            return group, min(sorted(labels & names), key=lambda name: counts[name])
    for sub, names in AMBIENT.items():
        if labels & names:
            return "ambient", sub
    return None


def select(fsd: str, cap_ambient: int, cap_hard: int, seed: int) -> list[dict]:
    rows = []
    counts = Counter()
    for split in ("dev", "eval"):
        for r in csv.DictReader(open(f"{fsd}/FSD50K.ground_truth/{split}.csv")):
            counts.update(r["labels"].split(","))
    for split in ("dev", "eval"):
        info = json.load(open(f"{fsd}/FSD50K.metadata/{split}_clips_info_FSD50K.json"))
        pools = defaultdict(list)
        for r in csv.DictReader(open(f"{fsd}/FSD50K.ground_truth/{split}.csv")):
            labels = set(r["labels"].split(","))
            got = assign(labels, counts)
            if got is None:
                continue
            group, sub = got
            meta = info[r["fname"]]
            row = {
                "fname": r["fname"],
                "split": split,
                "group": group,
                "subgroup": sub,
                "labels": sorted(labels),
                "title": meta["title"],
                "tags": meta["tags"],
                "license": meta["license"],
                "uploader": meta["uploader"],
            }
            if group == "doorbell":
                row["subgroup"] = doorbell_category(meta["title"], meta["tags"])
                if row["subgroup"] == "exclude":
                    continue
            if group in ("doorbell", "knock"):
                rows.append(row)
            else:
                pools[(group, sub)].append(row)
        rng = random.Random(f"{seed}-{split}")
        for key in sorted(pools):
            pool = pools[key]
            rng.shuffle(pool)
            rows += pool[: cap_ambient if key[0] == "ambient" else cap_hard]
    return rows


def _decode(args):
    path, wanted, out = args
    import pyarrow.parquet as pq
    import soundfile as sf

    done = {}
    table = pq.read_table(path)
    for name, audio in zip(
        table.column("name").to_pylist(), table.column("audio").to_pylist()
    ):
        split = name.split("/")[0].replace("_audio", "")
        fname = name.split("/")[1].rsplit("_", 1)[0]
        if (split, fname) not in wanted:
            continue
        x, fs = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        x = resample(x.mean(axis=1), fs, STORE_RATE)
        save_audio(out / "clips" / split / f"{fname}.npy", x)
        done[(split, fname)] = {
            "duration": round(len(x) / STORE_RATE, 3),
            "onset": round(onset_seconds(x, STORE_RATE), 3),
            "active_db": round(10 * np.log10(active_power(x, STORE_RATE)), 2),
        }
    return done


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--fsd50k",
        default=None,
        help="dir with FSD50K.ground_truth/, FSD50K.metadata/, parquet/",
    )
    ap.add_argument(
        "--cap-ambient",
        type=int,
        default=500,
        help="clips per ambient subgroup and split",
    )
    ap.add_argument(
        "--cap-hard", type=int, default=80, help="clips per hard label and split"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    root = data_root()
    fsd = args.fsd50k or str(root / "fsd50k")

    rows = select(fsd, args.cap_ambient, args.cap_hard, args.seed)
    wanted = {(r["split"], r["fname"]) for r in rows}
    shards = sorted(Path(fsd, "parquet").glob("*.parquet"))
    found = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for done in ex.map(_decode, [(p, wanted, root) for p in shards]):
            found.update(done)
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} selected clips are not in the parquet files")
    for r in rows:
        r.update(found[(r["split"], r["fname"])])
    write_jsonl(root / "clips.jsonl", rows)

    counts = Counter((r["split"], r["group"], r["subgroup"]) for r in rows)
    hours = defaultdict(float)
    for r in rows:
        hours[(r["split"], r["group"])] += r["duration"] / 3600
    for key in sorted(counts):
        print(*key, counts[key])
    for key in sorted(hours):
        print(*key, f"{hours[key]:.2f} h")


if __name__ == "__main__":
    main()
