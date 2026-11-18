#!/usr/bin/env python3
"""Offline evaluation of the GPSR planner (aha_tasks.gpsr_planner) without ROS.

Generates commands with the official command generator (a clone of
RoboCupAtHome/CommandGenerator, given by --generator; its code is imported,
not copied) over this arena's vocabulary: the names, rooms, locations and
objects of aha_tasks config/vocabulary/<task>.json are written as a generator
data directory. Each command is parsed by the grammar parser (the expected
steps), planned by the LLM, and compared. With --rephrase, the generator's own
rephrasing prompt (llm.py, as in its GPSR UI) asks the LLM for three
paraphrases of the first commands, which are planned and compared to the
original's expected steps. Batches of three commands (people, objects, any:
the generator's batch option) are interleaved to measure the travel saved and
the plan length.

Prints a summary; --output writes every command, plan and error as JSON.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "overlay_ws" / "src" / "aha_tasks"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE.parent / "aha_perception"))

from aha_tasks.core import load_spec  # noqa: E402
from aha_tasks.gpsr_planner import (  # noqa: E402
    Planner,
    complete_command,
    interleave,
    interleaved,
    plan_travel,
    sequential,
)
from aha_tasks.gpsr_vocabulary import (  # noqa: E402
    OPERATOR,
    load_vocabulary,
    names_of,
    spoken,
)
from aha_tasks.llm import DEFAULT_MODEL, DEFAULT_URL, Chat  # noqa: E402
from aha_tasks.plan import plan_slots  # noqa: E402

PARALLEL = 4  # requests the DGX server handles at once


def write_generator_data(vocabulary, directory):
    """The vocabulary as the generator's data directory (CompetitionTemplate layout)."""
    root = Path(directory)
    for sub in ("names", "maps", "objects"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    names = ["| Names |", "| --- |"] + [f"| {n} |" for n in vocabulary.names]
    (root / "names" / "names.md").write_text("\n".join(names) + "\n")
    rooms = ["| Name |", "| --- |"] + [f"| {spoken(r)} |" for r in vocabulary.rooms]
    (root / "maps" / "room_names.md").write_text("\n".join(rooms) + "\n")
    rows = ["| Number | Name | Object Category |", "| --- | --- | --- |"]
    labels = [(p, True) for p in vocabulary.placements]
    labels += [(b, False) for b in vocabulary.beacons]
    for number, (label, placement) in enumerate(labels, 1):
        mark = " (p)" if placement else ""
        rows.append(f"| {number} | {spoken(label)}{mark} | |")
    (root / "maps" / "location_names.md").write_text("\n".join(rows) + "\n")
    blocks = []
    for category in vocabulary.categories:
        plural = category.plural.replace(" ", "_")
        singular = category.singular.replace(" ", "_")
        blocks.append(f"# Class {plural} ({singular})")
        blocks.append("| Objectname | Image |")
        for label, name in vocabulary.object_categories.items():
            if name == category.name:
                blocks.append(f"| {label.replace(' ', '_')} | |")
        blocks.append("")
    (root / "objects" / "objects.md").write_text("\n".join(blocks))


def signature(steps):
    """Steps without words that only serve speech (people's descriptions, greetings)."""
    result = []
    for step in steps:
        params = dict(step["params"])
        if "person" in params and params["person"] != OPERATOR:
            params["person"] = "<person>"
        if step["action"] == "say":
            params["text"] = "<text>"
        result.append((step["action"], json.dumps(params, sort_keys=True)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator",
        type=Path,
        required=True,
        help="CommandGenerator clone (its src/ is imported)",
    )
    parser.add_argument("--count", type=int, default=60, help="commands to generate")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--rephrase", type=int, default=10, help="commands to rephrase (3 each)"
    )
    parser.add_argument("--task", default="gpsr")
    parser.add_argument("--llm-url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0, help="s per command")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.generator / "src"
    sys.path.insert(0, str(source if source.is_dir() else args.generator))
    from robocupathome_generator.gpsr_commands import CommandGenerator
    from robocupathome_generator.knowledge import parse_data

    spec = load_spec(PACKAGE / "config" / f"{args.task}.json")
    vocabulary = load_vocabulary(
        PACKAGE / "config" / "vocabulary" / f"{args.task}.json",
        spec.locations,
        names_of(spec.initial_state),
    )
    with tempfile.TemporaryDirectory() as data:
        write_generator_data(vocabulary, data)
        generator = CommandGenerator(parse_data(data))
    random.seed(args.seed)
    categories = ["people", "objects", ""]
    commands = []
    for i in range(args.count):
        text = generator.generate_command_start(cmd_category=categories[i % 3])
        commands.append(text[0].upper() + text[1:])

    chat = Chat(args.llm_url, args.model)
    planner = Planner(
        vocabulary,
        spec,
        chat,
        mode="llm",
        attempts=args.attempts,
        thinking=args.thinking,
    )
    grammar = Planner(vocabulary, spec, mode="grammar")
    expected = {}
    for text in commands:
        result = grammar.plan_command(1, text, time.monotonic() + 10)
        expected[text] = result.steps or None
        if not result.steps:
            print(f"GRAMMAR FAILED: {text}: {result.error}")

    cases = [(text, text) for text in commands]
    if args.rephrase:
        from robocupathome_generator.llm import SimpleOpenaiAPI

        api = SimpleOpenaiAPI(args.llm_url.rstrip("/") + "/chat/completions", "none")
        api.model = args.model
        for text in commands[: args.rephrase]:
            try:
                variants = api.alternativePhrasing(text)
            except Exception as error:
                print(f"REPHRASE FAILED: {text}: {error}")
                continue
            cases += [(variant.strip(), text) for variant in variants]

    def run(case):
        text, original = case
        started = time.monotonic()
        result = planner.plan_command(1, text, started + args.timeout)
        return {
            "text": text,
            "original": original,
            "rephrased": text != original,
            "valid": bool(result.steps),
            "correct": expected.get(original) is not None
            and signature(result.steps) == signature(expected[original]),
            "seconds": round(time.monotonic() - started, 2),
            "tries": len(result.attempts),
            "error": result.error,
            "steps": result.steps,
            "expected": expected.get(original),
        }

    with ThreadPoolExecutor(PARALLEL) as pool:
        rows = list(pool.map(run, cases))

    batches = []
    for start in range(0, len(commands) - 2, 3):
        texts = commands[start : start + 3]
        if any(expected[t] is None for t in texts):
            continue
        plans = {
            n: complete_command(grammar.parser.parse(t), n, vocabulary, spec)[0]
            for n, t in enumerate(texts, 1)
        }
        plan = interleave(plans, spec, vocabulary.operator, vocabulary)
        ends = (vocabulary.operator, vocabulary.operator)
        batches.append(
            {
                "commands": texts,
                "steps": len(plan),
                "travel": round(plan_travel(plan, spec, *ends), 2),
                "sequential_travel": round(
                    plan_travel(sequential(plans), spec, *ends), 2
                ),
                "interleaved": interleaved(plan),
            }
        )

    def rate(selected, key):
        if not selected:
            return "-"
        hits = sum(1 for row in selected if row[key])
        return f"{hits}/{len(selected)} ({100.0 * hits / len(selected):.0f} %)"

    covered = sum(1 for t in commands if expected[t] is not None)
    print(f"Grammar parses {covered}/{len(commands)} generated commands")
    for name, selected in (
        ("generated", [r for r in rows if not r["rephrased"]]),
        ("rephrased", [r for r in rows if r["rephrased"]]),
    ):
        seconds = sorted(r["seconds"] for r in selected) or [0.0]
        print(
            f"LLM {name}: valid {rate(selected, 'valid')}, "
            f"same steps as the grammar {rate(selected, 'correct')}, "
            f"median {seconds[len(seconds) // 2]:.1f} s, max {seconds[-1]:.1f} s, "
            f"retried {sum(1 for r in selected if r['tries'] > 1)}"
        )
    for row in rows:
        if not row["correct"]:
            print(f"- {'R' if row['rephrased'] else 'G'} {row['text']!r}")
            if row["rephrased"]:
                print(f"    original {row['original']!r}")
            if row["error"]:
                print(f"    error: {row['error']}")
            got = [f"{s['action']} {s['params']}" for s in row["steps"]]
            want = [f"{s['action']} {s['params']}" for s in row["expected"] or []]
            print(f"    got:      {got}")
            print(f"    expected: {want}")
    if batches:
        steps = sorted(b["steps"] for b in batches)
        saved = [b["sequential_travel"] - b["travel"] for b in batches]
        capacity = len(plan_slots(spec))
        print(
            f"Batches of 3: {len(batches)}, steps median {steps[len(steps) // 2]} "
            f"max {steps[-1]} (fit {capacity}: "
            f"{sum(1 for s in steps if s <= capacity)}), interleaved "
            f"{sum(1 for b in batches if b['interleaved'])}, travel saved "
            f"mean {sum(saved) / len(saved):.1f} m of "
            f"{sum(b['sequential_travel'] for b in batches) / len(batches):.1f} m"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"commands": rows, "batches": batches}, indent=1)
        )


if __name__ == "__main__":
    main()
