"""GPSR commands: grammar, planner (fake LLM), interleaving, skills and sim voices."""

import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from aha_tasks.core import Skill, parse_spec  # noqa: E402
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.gpsr_commands import (  # noqa: E402
    ListenCommand,
    PlanCommands,
    ReportResults,
    restated,
    yes_no,
)
from aha_tasks.gpsr_grammar import ParseError, Parser, normalize  # noqa: E402
from aha_tasks import gpsr_planner as gp  # noqa: E402
from aha_tasks.gpsr_vocabulary import load_vocabulary, names_of  # noqa: E402
from aha_tasks.interaction import NO_ASR  # noqa: E402
from aha_tasks.llm import Reply  # noqa: E402
from aha_tasks.plan import validate_plan  # noqa: E402
from aha_tasks.runner import load_task  # noqa: E402
from aha_tasks.sim_people import load_scene  # noqa: E402
from aha_tasks.sim_speech import (  # noqa: E402
    Speaker,
    Voices,
    asks_name,
    load_config,
    speaker_for,
)

GPSR = PACKAGE / "config" / "gpsr.json"
SPEC, _ = load_task(GPSR)
VOCABULARY = load_vocabulary("gpsr", SPEC.locations, names_of(SPEC.initial_state))
COLA = (
    "Get a cola from the kitchen table and bring it to the waving person in the office"
)
DRINKS = "Tell me how many drinks there are on the desk"
TIME = "Say the time to the person raising their left arm in the bedroom"


class NoInfo(Skill):
    """A find_described_person that does not take person info."""

    @classmethod
    def validate(cls, params, spec):
        if "info" in params:
            raise ValueError("unknown info")


def spec_with(find_described_person):
    data = json.loads(GPSR.read_text())
    data["skills"]["find_described_person"] = find_described_person
    return parse_spec(data)


def nav(location):
    return {"action": "navigate", "params": {"location": location}}


def act(action, **params):
    return {"action": action, "params": params}


def actions(steps):
    return [(s["action"], s["params"]) for s in steps]


def complete(steps, number=1):
    return gp.complete_command(steps, number, VOCABULARY, SPEC)


class FakeChat:
    """Replies with the given contents in turn and records the conversations."""

    def __init__(self, *contents, error=None):
        self.contents = list(contents)
        self.error = error
        self.calls = []

    def __call__(self, messages, schema=None, **kwargs):
        self.calls.append({"messages": messages, "schema": schema, **kwargs})
        if self.error is not None:
            raise self.error
        content = self.contents.pop(0)
        if not isinstance(content, str):
            content = json.dumps({"steps": content})
        return Reply(content, "", "stop", {}, 0.1)


def check_schema(value, schema, path="$"):
    """Errors of value against the subset of JSON schema that the planner uses."""
    if "anyOf" in schema:
        options = [check_schema(value, option, path) for option in schema["anyOf"]]
        return [] if any(not errors for errors in options) else [f"{path}: no match"]
    if "const" in schema:
        return [] if value == schema["const"] else [f"{path}: not {schema['const']}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: {value!r} not in enum"]
    kind = schema.get("type")
    if kind == "string":
        return [] if isinstance(value, str) else [f"{path}: not a string"]
    if kind == "array":
        if not isinstance(value, list):
            return [f"{path}: not an array"]
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1e9):
            return [f"{path}: length"]
        return [
            e
            for i, v in enumerate(value)
            for e in check_schema(v, schema["items"], f"{path}[{i}]")
        ]
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path}: not an object"]
        errors = [f"{path}: missing {k}" for k in schema["required"] if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [
                f"{path}: extra {k}" for k in value if k not in schema["properties"]
            ]
        for key, item in value.items():
            if key in schema["properties"]:
                errors += check_schema(item, schema["properties"][key], f"{path}.{key}")
        return errors
    return []


class VocabularyTests(unittest.TestCase):
    def test_arena_words(self):
        self.assertIn("Charlie", VOCABULARY.names)
        self.assertEqual(VOCABULARY.label("kitchen table"), "kitchen_table")
        self.assertIsNone(VOCABULARY.label("bathroom"))
        self.assertEqual(VOCABULARY.category("drinks"), "drink")
        self.assertEqual(VOCABULARY.category("cola"), "drink")
        hotwords = VOCABULARY.hotwords()
        for word in ("Charlie", "kitchen table", "living room", "cola", "drinks"):
            self.assertIn(word, hotwords)

    def test_locations_must_belong_to_the_task(self):
        with self.assertRaises(ValueError):
            load_vocabulary("gpsr", {"bedroom": [0, 0, 0]})


class GrammarTests(unittest.TestCase):
    parser = Parser(VOCABULARY)

    def test_take_and_deliver_to_a_person(self):
        self.assertEqual(
            self.parser.parse(COLA),
            [
                nav("kitchen_table"),
                act("pick_object", object="cola", surface="kitchen_table"),
                nav("office"),
                act(
                    "find_described_person", gesture="waving person", location="office"
                ),
                act("deliver_object", object="cola", person="waving person"),
            ],
        )

    def test_questions_end_at_the_operator(self):
        self.assertEqual(
            self.parser.parse(DRINKS + "."),
            [
                nav("desk"),
                act("count", target="drinks", location="desk"),
                nav("instruction_point"),
                act("tell_info", person="operator"),
            ],
        )

    def test_follow_ups_chain(self):
        steps = self.parser.parse(
            "Go to the living room then find a snack and take it and throw it in the trash"
        )
        self.assertEqual(
            steps,
            [
                nav("living_room"),
                act("find_object", object="snack", location="living_room"),
                act("pick_object", object="snack"),
                nav("trashbin"),
                act("place_object", object="snack", surface="trashbin"),
            ],
        )

    def test_names_keep_their_spelling(self):
        steps = self.parser.parse("greet charlie in the kitchen and follow them")
        self.assertEqual(steps[1]["params"]["name"], "Charlie")
        self.assertEqual(steps[2], act("say", text="Hello Charlie, I am AhaRobot."))
        self.assertEqual(steps[3], act("follow_person", person="Charlie"))

    def test_person_info_is_found_and_told(self):
        self.assertEqual(
            self.parser.parse(
                "Tell the gesture of the person at the bed to the person at the desk"
            ),
            [
                nav("bed"),
                act("find_described_person", info="gesture", location="bed"),
                nav("desk"),
                act("tell_info", person="person at the desk", location="desk"),
            ],
        )
        steps = self.parser.parse("Tell me the name of the person in the kitchen")
        self.assertEqual(
            steps[1], act("find_described_person", info="name", location="kitchen")
        )
        self.assertEqual(steps[3], act("tell_info", person="operator"))

    def test_clothing_and_guiding(self):
        steps = self.parser.parse(
            "Lead the person wearing an orange T-shirt from the entrance to the bed"
        )
        self.assertEqual(
            steps[1]["params"],
            {"clothing": "person wearing an orange t shirt", "location": "entrance"},
        )
        self.assertEqual(steps[2]["params"]["location"], "bed")

    def test_rephrased_or_unknown_commands_are_left_to_the_llm(self):
        for text in (
            "Could you grab the cola from the kitchen table for me?",
            "Bring me a cola from the bathroom sink",
            "",
        ):
            with self.assertRaises(ParseError):
                self.parser.parse(text)

    def test_normalize(self):
        self.assertEqual(
            normalize("The first command is: Tell your team's name!"),
            "tell your teams name",
        )

    def test_every_example_fits_the_arena(self):
        messages = gp.few_shot(VOCABULARY)
        self.assertEqual(len(messages), 2 * len(gp.EXAMPLES))


class CompleteCommandTests(unittest.TestCase):
    def test_talking_needs_the_person_found_before(self):
        for action, params in (
            ("answer_question", {"question": "the time"}),
            ("say", {"text": "Hello"}),
        ):
            with self.subTest(action):
                _, errors = complete([act(action, **params)])
                self.assertTrue(errors)
                find = act(
                    "find_described_person", gesture="waving person", location="office"
                )
                _, errors = complete([find, act(action, **params)])
                self.assertEqual(errors, [])

    def test_tell_info_needs_someone_to_tell(self):
        count = act("count", target="drinks", location="desk")
        for person, extra, ok in (
            ("operator", {}, True),
            ("Jane", {"location": "bed"}, True),
            ("me", {}, False),
            ("Jane", {}, False),
        ):
            with self.subTest(person=person, extra=extra):
                _, errors = complete([count, act("tell_info", person=person, **extra)])
                self.assertEqual(errors == [], ok, errors)

    def test_adds_navigation_keys_and_mocks(self):
        steps, errors = complete(
            [
                act("find_object", object="apple", location="kitchen"),
                act("pick_object", object="apple"),
                act("deliver_object", object="apple", person="operator"),
            ],
            number=2,
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            actions(steps),
            [
                ("navigate", {"location": "kitchen"}),
                (
                    "find_object",
                    {"object": "apple", "location": "kitchen", "output": "cmd2.object"},
                ),
                ("pick_object", {"object": "cmd2.object"}),
                ("navigate", {"location": "instruction_point"}),
                ("deliver_object", {"object": "apple", "person": "operator"}),
            ],
        )
        self.assertEqual(steps[2]["mock"], {"robot.holding": "apple"})
        self.assertEqual(steps[4]["mock"], {"robot.holding": None})
        self.assertTrue(all(s["command"] == 2 for s in steps))

    def test_answers_get_keys_and_repeated_navigation_is_dropped(self):
        steps, errors = complete(
            [
                nav("desk"),
                nav("desk"),
                act("describe", target="biggest object", location="desk"),
                nav("office"),
                nav("instruction_point"),
                act("tell_info", person="operator"),
            ]
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            [s["action"] for s in steps],
            ["navigate", "describe", "navigate", "tell_info"],
        )
        self.assertEqual(steps[1]["params"]["output"], "cmd1.info")
        self.assertEqual(steps[3]["params"]["key"], "cmd1.info")

    def test_person_info_has_a_role_of_the_command(self):
        steps = [
            nav("bed"),
            act("find_described_person", info="pose", location="bed"),
            act("tell_info", person="person at the desk", location="desk"),
        ]
        spec = spec_with(None)  # any find_described_person params pass
        plan, errors = gp.complete_command(steps, 3, VOCABULARY, spec)
        self.assertEqual(errors, [])
        self.assertEqual(plan[1]["params"]["role"], "cmd3.person")
        self.assertEqual(
            [s["action"] for s in plan],
            ["navigate", "find_described_person", "navigate", "tell_info"],
        )
        self.assertEqual(plan[3]["params"]["key"], "cmd3.person.pose")
        self.assertEqual(gp.answer_key(plan[1]), "cmd3.person.pose")

    def test_person_info_is_described_without_info_in_the_search(self):
        steps = [
            nav("bed"),
            act("find_described_person", info="pose", location="bed"),
            nav("instruction_point"),
            act("tell_info", person="operator"),
        ]
        spec = spec_with(f"{__name__}:NoInfo")
        plan, errors = gp.complete_command(steps, 3, VOCABULARY, spec)
        self.assertEqual(errors, [])
        self.assertEqual(
            plan[1],
            {
                "command": 3,
                "action": "describe",
                "params": {
                    "target": "pose of the person",
                    "location": "bed",
                    "output": "cmd3.info",
                },
            },
        )
        self.assertEqual(plan[3]["params"]["key"], "cmd3.info")

    def test_guiding_moves_the_robot(self):
        steps, errors = complete(
            [
                nav("bed"),
                act("find_described_person", name="Jane", location="bed"),
                act("guide_person", person="Jane", location="office"),
            ]
        )
        self.assertEqual(errors, [])
        self.assertEqual(steps[2]["mock"], {"robot.location": "office"})

    def test_errors(self):
        cases = [
            ([act("pick_object", object="cola")], "needs a surface"),
            (
                [
                    act("pick_object", object="cola", surface="desk"),
                    act("pick_object", object="milk", surface="desk"),
                    act("place_object", object="milk", surface="bed"),
                ],
                "still holds the cola",
            ),
            (
                [act("place_object", object="cola", surface="bed")],
                "without a held object",
            ),
            (
                [act("pick_object", object="cola", surface="desk")],
                "still held at the end",
            ),
            ([act("tell_info", person="operator")], "needs a count"),
            ([act("follow_person", person="them")], "find the person"),
            (
                [
                    act("pick_object", object="cola", surface="desk"),
                    act("deliver_object", object="cola", person="Jane"),
                ],
                "find the person",
            ),
            ([act("teleport")], "unknown action"),
            ([act("run_plan_step", index=0)], "unknown action"),
            ([nav("bathroom")], "Unknown location"),
            ([nav("desk")], "other than navigate"),
            ([], "at least one step"),
        ]
        for steps, text in cases:
            with self.subTest(text):
                _, errors = complete(steps)
                self.assertTrue(any(text in e for e in errors), errors)


class SchemaAndPromptTests(unittest.TestCase):
    def test_examples_satisfy_the_schema(self):
        schema = gp.command_schema(VOCABULARY, gp.plannable(SPEC))
        messages = gp.few_shot(VOCABULARY)
        for message in messages[1::2]:
            data = json.loads(message["content"])
            self.assertEqual(check_schema(data, schema), [], message["content"])

    def test_schema_rejects_words_outside_the_arena(self):
        schema = gp.command_schema(VOCABULARY, gp.plannable(SPEC))
        for step in (
            nav("bathroom"),
            act("pick_object", object="banana", surface="desk"),
            act("find_described_person", gesture="dancing person", location="office"),
            act("plan_commands", commands=[]),
        ):
            with self.subTest(step):
                self.assertNotEqual(check_schema({"steps": [step]}, schema), [])

    def test_prompt_lists_the_arena(self):
        prompt = gp.system_prompt(VOCABULARY, gp.plannable(SPEC))
        for word in (
            "kitchen_table",
            "Charlie",
            "cola",
            "drink / drinks",
            "instruction_point",
        ):
            self.assertIn(word, prompt)
        self.assertNotIn("run_plan_step", prompt)


class PlannerTests(unittest.TestCase):
    def planner(self, chat, mode="llm"):
        return gp.Planner(VOCABULARY, SPEC, chat, mode=mode)

    def test_canonical_commands_need_no_llm(self):
        chat = FakeChat()
        plan, results = self.planner(chat, "auto").plan(
            {1: COLA, 2: DRINKS, 3: TIME}, 10
        )
        self.assertEqual(chat.calls, [])
        self.assertEqual([r.planner for r in results], ["grammar"] * 3)
        validate_plan(plan, SPEC)
        self.assertEqual({s["command"] for s in plan}, {1, 2, 3})

    def test_llm_steps_with_the_schema(self):
        chat = FakeChat(
            [
                nav("desk"),
                act("count", target="drinks", location="desk"),
                act("tell_info", person="operator"),
            ]
        )
        plan, results = self.planner(chat).plan(
            {2: "How many drinks are on the desk?"}, 10
        )
        self.assertEqual(results[0].planner, "llm")
        self.assertEqual(
            plan[-1]["params"], {"key": "cmd2.count", "person": "operator"}
        )
        call = chat.calls[0]
        self.assertEqual(call["messages"][0]["role"], "system")
        self.assertEqual(
            call["messages"][-1]["content"], "How many drinks are on the desk?"
        )
        self.assertIn("anyOf", json.dumps(call["schema"]))
        self.assertFalse(call["thinking"])

    def test_errors_are_sent_back_for_another_try(self):
        chat = FakeChat(
            "not json",
            [act("tell_info", person="operator")],
            [
                nav("desk"),
                act("count", target="drinks", location="desk"),
                act("tell_info", person="operator"),
            ],
        )
        _, results = self.planner(chat).plan({1: "drinks on the desk, how many?"}, 10)
        result = results[0]
        self.assertEqual(result.planner, "llm")
        self.assertEqual(len(result.attempts), 3)
        feedback = chat.calls[2]["messages"][-1]["content"]
        self.assertIn("needs a count", feedback)
        self.assertEqual(chat.calls[2]["messages"][-2]["role"], "assistant")

    def test_failures_after_the_attempts(self):
        chat = FakeChat(*([[act("tell_info", person="operator")]] * 3))
        plan, results = self.planner(chat).plan({1: "tell me"}, 10)
        self.assertEqual(plan, [])
        self.assertEqual(results[0].planner, "")
        self.assertIn("needs a count", results[0].error)

    def test_unreachable_llm_and_grammar_fallback(self):
        chat = FakeChat(error=OSError("connection refused"))
        plan, results = self.planner(chat, "auto").plan(
            {1: COLA, 2: "Rephrased, so the grammar fails"}, 10
        )
        self.assertEqual(results[0].planner, "grammar")
        self.assertIn("connection refused", results[1].error)
        self.assertEqual({s["command"] for s in plan}, {1})

    def test_no_request_after_the_time_budget(self):
        chat = FakeChat()
        _, results = self.planner(chat).plan({1: "anything"}, 0.5)
        self.assertEqual(chat.calls, [])
        self.assertIn("out of time", results[0].error)


class InterleaveTests(unittest.TestCase):
    def plans(self, *texts):
        parser = Parser(VOCABULARY)
        return {n: complete(parser.parse(t), n)[0] for n, t in enumerate(texts, 1)}

    def test_shorter_travel_than_one_by_one(self):
        plans = self.plans(COLA, DRINKS, TIME)
        plan = gp.interleave(plans, SPEC, "instruction_point", VOCABULARY)
        validate_plan(plan, SPEC)
        ends = ("instruction_point", "instruction_point")
        self.assertLess(
            gp.plan_travel(plan, SPEC, *ends),
            gp.plan_travel(gp.sequential(plans), SPEC, *ends) - 1.0,
        )
        self.assertTrue(gp.interleaved(plan))
        for n in plans:  # each command keeps its order
            self.assertEqual([s for s in plan if s["command"] == n], plans[n])

    def test_one_object_at_a_time(self):
        plans = self.plans(
            "Bring me a cola from the kitchen table",
            "Bring me a milk from the side tables",
            "Take an apple from the kitchen table and put it on the side tables",
        )
        plan = gp.interleave(plans, SPEC, "instruction_point", VOCABULARY)
        holding = None
        for step in plan:
            if step["action"] == "pick_object":
                self.assertIsNone(holding)
                holding = step["command"]
            elif step["action"] in ("place_object", "deliver_object"):
                holding = None

    def test_no_pick_while_another_command_holds_within_a_block(self):
        # Command 1 picks and gives away within one block (one room); command 2
        # holds the cola between its blocks.
        plans = self.plans(
            "Find a sponge in the kitchen then take it and give it to the waving "
            "person in the kitchen",
            "Get a cola from the kitchen table and deliver it to the waving person "
            "in the office",
        )
        plan = gp.interleave(plans, SPEC, "instruction_point", VOCABULARY)
        holding = None
        for step in plan:
            if step["action"] == "pick_object":
                self.assertIsNone(holding, plan)
                holding = step["command"]
            elif step["action"] in ("place_object", "deliver_object"):
                holding = None

    def test_equal_travel_keeps_the_command_order(self):
        plans = self.plans(TIME, TIME)
        plan = gp.interleave(plans, SPEC, "instruction_point", VOCABULARY)
        self.assertEqual([s["command"] for s in plan], [1, 1, 1, 2, 2])
        self.assertFalse(gp.interleaved(plan))

    def test_both_answers_are_told_after_one_trip(self):
        plans = self.plans(DRINKS, "Tell me how many snacks there are on the desk")
        plan = gp.interleave(plans, SPEC, "instruction_point", VOCABULARY)
        self.assertEqual(
            [s["action"] for s in plan],
            ["navigate", "count", "count", "navigate", "tell_info", "tell_info"],
        )
        self.assertTrue(gp.interleaved(plan))

    def test_navigation_to_where_the_robot_is_is_dropped(self):
        plans = self.plans(DRINKS, "Tell me the name of the person at the desk")
        plan = gp.interleave(plans, SPEC, "instruction_point", VOCABULARY)
        self.assertEqual(
            sum(1 for s in plan if s["params"].get("location") == "desk"), 3
        )


class WordsTests(unittest.TestCase):
    def test_speech_and_picture(self):
        parser = Parser(VOCABULARY)
        steps, _ = complete(parser.parse(COLA))
        self.assertEqual(
            gp.command_summary(steps, VOCABULARY),
            "pick up the cola from the kitchen table, find the waving person in the "
            "office and give the cola to the waving person",
        )
        speech = gp.plan_speech(steps, VOCABULARY)
        self.assertTrue(speech.startswith("First, for command 1, I will pick up"))
        image = gp.render_plan(steps, VOCABULARY)
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")

    def test_report(self):
        plans = {1: complete(Parser(VOCABULARY).parse(DRINKS))[0]}
        plan = plans[1]
        status = {
            "subgoals": [{"skill": "say", "status": "succeeded"}]
            + [{"skill": "run_plan_step", "status": "succeeded", "message": "ok"}] * 3
            + [
                {
                    "skill": "run_plan_step",
                    "status": "succeeded",
                    "message": "tell_info is not implemented; mock state assumed",
                }
            ]
        }
        outcomes = gp.step_outcomes(status, plan)
        self.assertEqual(outcomes, {0: "done", 1: "done", 2: "done", 3: "mocked"})
        commands = [{"command": 1, "summary": gp.command_summary(plan, VOCABULARY)}]
        memory = {"cmd1.count": 3}
        sentences = gp.report_sentences(plan, commands, outcomes, memory)
        self.assertIn("not finished, 3 of 4 steps done", sentences[0])
        self.assertIn("The answer was 3.", sentences[0])
        outcomes[3] = "done"
        memory["cmd1.count.answer"] = "There are 3 drinks."
        sentences = gp.report_sentences(plan, commands, outcomes, memory)
        self.assertTrue(sentences[0].endswith(": done. There are 3 drinks."))
        missing = [{"command": 2, "summary": ""}]
        self.assertEqual(
            gp.report_sentences(plan, missing, {}, {}),
            ["Command 2: I did not understand it."],
        )
        dropped = [{"command": 3, "summary": "", "dropped": True}]
        self.assertEqual(
            gp.report_sentences(plan, dropped, {}, {}),
            ["Command 3: I understood it, but it did not fit my plan."],
        )


class ConfirmationTests(unittest.TestCase):
    def test_yes_no(self):
        for answer, verdict in (
            ("Yes, that's it.", "yes"),
            ("That's right", "yes"),
            ("correct", "yes"),
            ("Sure, why not", "yes"),
            ("No.", "no"),
            ("uh, no", "no"),
            ("That's not right", "no"),
            ("Not quite", "no"),
            ("", None),
            ("What?", None),
            ("I didn't hear you", None),
            ("I'm not sure", None),
            ("Not sure", None),
            ("Absolutely not", "no"),
            ("Certainly not.", "no"),
        ):
            with self.subTest(answer):
                self.assertEqual(yes_no(answer)[0], verdict)
        self.assertEqual(
            yes_no("Bring me a cola from the desk"),
            (None, "bring me a cola from the desk"),
        )

    def test_restated(self):
        for answer in (
            "No, no, sorry, bring me a cola from the desk",
            "Sorry, no, I said bring me a cola from the desk",
            "Um, no, I said bring me a cola from the desk",
            "Oh no, I said bring me a cola from the desk",
            "That's not right, I said bring me a cola from the desk",
            "Not quite, I said bring me a cola from the desk",
        ):
            with self.subTest(answer):
                verdict, rest = yes_no(answer)
                self.assertEqual(verdict, "no")
                self.assertEqual(
                    restated(rest, VOCABULARY), "bring me a cola from the desk"
                )
        verdict, rest = yes_no("No, I said bring me a cola from the desk")
        self.assertEqual(verdict, "no")
        self.assertEqual(restated(rest, VOCABULARY), "bring me a cola from the desk")
        for rest in (
            "not that",
            "i didn't hear you at all",
            "pardon could you say that again",
            "could you say it again",
            "i didn't hear it",
            "can you tell me again please",
        ):
            with self.subTest(rest):
                self.assertIsNone(restated(rest, VOCABULARY))
        self.assertEqual(
            restated("go to the kitchen table and count the snacks", VOCABULARY),
            "go to the kitchen table and count the snacks",
        )


class FakeInteraction:
    """Says and shows; listen answers in turn (None is silence, NO_ASR no server)."""

    def __init__(self, *answers):
        self.said, self.shown, self.hidden = [], [], 0
        self.answers = list(answers)
        self.hotwords = []

    def say(self, text, cancelled):
        self.said.append(text)
        return True, "said"

    def show(self, image, image_format, text, cancelled):
        self.shown.append((image_format, len(image)))
        return self.say(text, cancelled)

    def hide(self):
        self.hidden += 1

    def listen(self, hotwords, cancelled, timeout=10.0, language="en"):
        self.hotwords.append(list(hotwords))
        answer = self.answers.pop(0)
        if answer is NO_ASR:
            return None, f"{NO_ASR}: /aha/perception/audio/listen is not served"
        if answer is None:
            return None, "ASR: no transcript within 10.0s"
        return answer, ""


def runtime(*answers):
    return SimpleNamespace(node=None, spec=SPEC, interaction=FakeInteraction(*answers))


def run_skill(skill, params, memory=None):
    ListenCommand.validate(params, SPEC) if isinstance(skill, ListenCommand) else None
    return skill.execute(
        SkillContext(
            subgoal="x",
            params=params,
            memory=memory or {},
            cancelled=threading.Event(),
            log=lambda text: None,
        )
    )


class ListenCommandTests(unittest.TestCase):
    def test_heard_and_confirmed(self):
        rt = runtime(COLA, "yes")
        result = run_skill(ListenCommand(rt), {"command": 1})
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs, {"cmd1.text": COLA})
        said = rt.interaction.said
        self.assertEqual(said[0], "Please tell me the first command.")
        self.assertIn(f"I heard: {COLA}.", said[1])
        self.assertIn("cola", rt.interaction.hotwords[0])
        self.assertEqual(rt.interaction.hotwords[1], ["yes", "no"])

    def test_silence_short_and_no_ask_again(self):
        rt = runtime(None, "cola", DRINKS, "no", DRINKS, "yeah")
        result = run_skill(ListenCommand(rt), {"command": 2})
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs, {"cmd2.text": DRINKS})
        self.assertIn("3 rephrasing request(s)", result.message)

    def test_correction_in_the_answer(self):
        rt = runtime(DRINKS, "No, I said " + TIME, "yes")
        result = run_skill(ListenCommand(rt), {"command": 3})
        self.assertEqual(result.outputs, {"cmd3.text": TIME.lower()})

    def test_a_last_correction_is_not_taken_unconfirmed(self):
        corrections = ["No, I said " + TIME] * 4
        rt = runtime(DRINKS, *corrections, None, None, None, None, None)
        result = run_skill(
            ListenCommand(rt), {"command": 3, "attempts": 1, "custom_operator": 0}
        )
        self.assertFalse(result.success)

    def test_a_correction_needs_a_yes(self):
        rt = runtime(DRINKS, None, "No, I said " + TIME, None, None, None, None)
        result = run_skill(
            ListenCommand(rt), {"command": 3, "attempts": 1, "custom_operator": 0}
        )
        self.assertFalse(result.success)

    def test_an_answer_that_is_neither_yes_nor_no_is_not_taken(self):
        rt = runtime(DRINKS, "I don't think so", "Hmm, I don't think that's right")
        rt.interaction.answers += ["I didn't hear you", None] + [None] * 8
        result = run_skill(
            ListenCommand(rt), {"command": 1, "attempts": 1, "custom_operator": 0}
        )
        self.assertFalse(result.success)

    def test_two_silent_confirmations_take_the_command(self):
        rt = runtime(DRINKS, None, None)
        result = run_skill(ListenCommand(rt), {"command": 1})
        self.assertEqual(result.outputs, {"cmd1.text": DRINKS})

    def test_no_asr_fails_at_once(self):
        rt = runtime(NO_ASR)
        result = run_skill(ListenCommand(rt), {"command": 1})
        self.assertFalse(result.success)
        self.assertTrue(result.message.startswith(NO_ASR))

    def test_custom_operator_after_the_referee(self):
        rt = runtime(None, None, None, None)
        result = run_skill(
            ListenCommand(rt), {"command": 1, "attempts": 3, "custom_operator": 1}
        )
        self.assertFalse(result.success)
        self.assertIn("custom operator", rt.interaction.said[-1])

    def test_params(self):
        for params in (
            {},
            {"command": 4},
            {"command": "1"},
            {"command": 1, "attempts": 0},
            {"command": 1, "x": 1},
        ):
            with self.subTest(params), self.assertRaises(ValueError):
                ListenCommand.validate(params, SPEC)


class PlanCommandsTests(unittest.TestCase):
    def skill(self, chat):
        rt = runtime()
        skill = PlanCommands(rt)
        skill.planner = lambda params: gp.Planner(
            VOCABULARY, SPEC, chat, mode=params.get("planner", "auto")
        )
        return skill, rt

    def test_plan_is_shown_and_said(self):
        skill, rt = self.skill(FakeChat())
        params = {"commands": ["cmd1.text", "cmd2.text", "cmd3.text"]}
        PlanCommands.validate(params, SPEC)
        memory = {
            "cmd1.text": COLA,
            "cmd2.text": DRINKS,
            "cmd3.text": TIME,
            "robot.location": "instruction_point",
        }
        result = run_skill(skill, params, memory)
        self.assertTrue(result.success, result.message)
        validate_plan(result.outputs["plan"], SPEC)
        self.assertEqual(
            [c["planner"] for c in result.outputs["plan.commands"]], ["grammar"] * 3
        )
        self.assertEqual(rt.interaction.shown[0][0], "png")
        speech = rt.interaction.said[0]
        self.assertIn("I understood commands 1, 2 and 3.", speech)
        self.assertIn("interleave", speech)

    def test_a_command_that_cannot_be_planned_is_said(self):
        skill, rt = self.skill(FakeChat(error=OSError("down")))
        memory = {"cmd1.text": COLA, "cmd2.text": "gibberish"}
        result = run_skill(
            skill, {"commands": ["cmd1.text", "cmd2.text", "cmd3.text"]}, memory
        )
        self.assertTrue(result.success)
        self.assertIn(
            "I could not understand commands 2 and 3.", rt.interaction.said[0]
        )

    def test_a_skipped_listening_is_not_planned_from_its_mock_text(self):
        skill, rt = self.skill(FakeChat())
        statuses = {"cmd1.listen": "succeeded", "cmd2.listen": "skipped"}
        skill.status = json.dumps(
            {"subgoals": [{"id": i, "status": s} for i, s in statuses.items()]}
        )
        memory = {"cmd1.text": COLA, "cmd2.text": DRINKS}
        result = run_skill(skill, {"commands": ["cmd1.text", "cmd2.text"]}, memory)
        self.assertTrue(result.success, result.message)
        self.assertEqual({s["command"] for s in result.outputs["plan"]}, {1})
        self.assertIn("I could not understand command 2.", rt.interaction.said[0])
        # The report still has a word about command 2.
        infos = result.outputs["plan.commands"]
        self.assertEqual([i["command"] for i in infos], [1, 2])
        self.assertEqual((infos[1]["summary"], infos[1]["error"]), ("", "not heard"))
        # Not selected (bypassed): the mock text is planned, e.g. for a test run.
        skill.status = json.dumps(
            {"subgoals": [{"id": "cmd2.listen", "status": "bypassed"}]}
        )
        result = run_skill(skill, {"commands": ["cmd1.text", "cmd2.text"]}, memory)
        self.assertEqual({s["command"] for s in result.outputs["plan"]}, {1, 2})

    def test_nothing_planned_fails(self):
        skill, rt = self.skill(FakeChat(error=OSError("down")))
        result = run_skill(
            skill, {"commands": ["cmd1.text"]}, {"cmd1.text": "gibberish"}
        )
        self.assertFalse(result.success)

    def test_params(self):
        for params in (
            {},
            {"commands": []},
            {"commands": ["a"] * 4},
            {"commands": ["a"], "planner": "x"},
        ):
            with self.subTest(params), self.assertRaises(ValueError):
                PlanCommands.validate(params, SPEC)


class ReportResultsTests(unittest.TestCase):
    def test_report_without_status_uses_the_plan(self):
        rt = runtime()
        skill = ReportResults(rt)
        skill.status = None
        plan = next(s for s in SPEC.subgoals if s.id == "plan.generate").mock["plan"]
        result = run_skill(skill, {}, {"plan": plan, "cmd2.count": 3})
        self.assertTrue(result.success, result.message)
        said = rt.interaction.said[0]
        self.assertTrue(
            said.startswith("Here is my report. Command 1, pick up the cola")
        )
        self.assertIn("The answer was 3.", said)
        self.assertEqual(rt.interaction.hidden, 1)

    def test_a_skipped_planning_reports_no_mock_commands(self):
        rt = runtime()
        skill = ReportResults(rt)
        plan = next(s for s in SPEC.subgoals if s.id == "plan.generate").mock["plan"]
        skill.status = json.dumps(
            {"subgoals": [{"id": "plan.generate", "status": "skipped"}]}
        )
        result = run_skill(skill, {}, {"plan": plan})
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            rt.interaction.said[0],
            "Here is my report. I could not plan the commands, so I did not do them.",
        )

    def test_status_gives_the_outcomes(self):
        rt = runtime()
        skill = ReportResults(rt)
        plan = next(s for s in SPEC.subgoals if s.id == "plan.generate").mock["plan"]
        slots = [{"skill": "run_plan_step", "status": "succeeded", "message": ""}] * 20
        skill.status = json.dumps({"subgoals": slots})
        result = run_skill(skill, {}, {"plan": plan})
        self.assertEqual(result.message.count(": done."), 3)


class SimSpeechTests(unittest.TestCase):
    def setUp(self):
        people = load_scene(PACKAGE / "config" / "sim" / "people" / "gpsr.json")
        self.settings, self.speakers = load_config(
            PACKAGE / "config" / "sim" / "speech" / "gpsr.json", SPEC.locations, people
        )
        self.voices = Voices(self.settings)

    def speaker(self, x, y, yaw):
        return speaker_for(
            (x, y, yaw), self.speakers, self.settings["radius"], self.settings["fov"]
        )

    def test_who_answers(self):
        operator = self.speaker(*SPEC.locations["instruction_point"])
        self.assertTrue(operator.operator)
        # Jane raises her left arm at (-2.0, 3.0) in the bedroom.
        self.assertEqual(self.speaker(-2.6, 2.4, 0.8).name, "Jane")
        self.assertIsNone(self.speaker(-2.6, 2.4, -2.3))  # facing away
        self.assertIsNone(self.speaker(3.0, -2.0, 0.0))  # nobody near

    def test_the_person_in_front_answers_the_name_not_the_operator(self):
        # Close to Charlie (-0.9, 2.8) but closer to the instruction point.
        pose = (-1.68, 2.17, 0.68)
        self.assertTrue(asks_name("Excuse me, are you Charlie?"))
        self.assertFalse(asks_name("Please tell me the first command."))
        charlie = speaker_for(
            pose,
            self.speakers,
            self.settings["radius"],
            self.settings["fov"],
            person=True,
        )
        self.assertEqual(charlie.name, "Charlie")
        operator = speaker_for(
            pose,
            self.speakers,
            self.settings["radius"],
            self.settings["fov"],
            person=False,
        )
        self.assertTrue(operator.operator)

    def test_commands_get_simpler_when_asked_again(self):
        operator = Speaker("", (0.0, 0.0), True)
        first = self.voices.answer("Please tell me the first command.", operator)
        again = self.voices.answer(
            "Sorry. Please tell me the first command again.", operator
        )
        self.assertEqual(first, self.settings["commands"][0][0])
        self.assertEqual(again, self.settings["commands"][0][1])
        third = self.voices.answer(
            "Could a custom operator please tell me the third command?", operator
        )
        self.assertEqual(third, self.settings["commands"][2][0])
        self.assertEqual(
            self.voices.answer(
                "I heard: x. Is that correct? Please say yes or no.", operator
            ),
            "yes",
        )
        self.assertIsNone(self.voices.answer("Hello!", operator))
        self.assertIsNone(self.voices.answer("Tell me the first command.", None))
        self.voices.new_run(2)
        first = self.voices.answer("Please tell me the first command.", operator)
        self.assertEqual(first, self.settings["commands"][0][0])

    def test_names(self):
        jane = Speaker("Jane", (0.0, 0.0), False)
        self.assertEqual(
            self.voices.answer("What is your name?", jane), "My name is Jane."
        )
        self.assertEqual(self.voices.answer("Are you Jane?", jane), "Yes.")
        self.assertEqual(
            self.voices.answer("Excuse me, are you Charlie?", jane),
            "No, my name is Jane.",
        )
        self.assertIsNone(self.voices.answer("Please tell me the first command.", jane))

    def test_bad_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.json"
            for config in (
                {"radius": 1.0},
                {"operator": {"location": "moon"}},
                {"operator": {"location": "instruction_point", "commands": [[]]}},
                {"operator": {"location": "instruction_point", "volume": 3}},
            ):
                path.write_text(json.dumps(config))
                with self.subTest(config), self.assertRaises(ValueError):
                    load_config(path, SPEC.locations)

    def test_people_scene_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.json"
            person = {
                "label": "standing",
                "shirt": "red",
                "position": [0, 0, 0],
                "yaw": 0,
                "name": "",
            }
            path.write_text(
                json.dumps({"world": "w", "wave_period": 0.8, "people": {"a": person}})
            )
            with self.assertRaises(ValueError):
                load_scene(path)


if __name__ == "__main__":
    unittest.main()
