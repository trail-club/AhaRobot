"""Appearance phrases and describe_person (no ROS graph)."""

import os
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.appearance import (  # noqa: E402
    choose_features,
    consensus,
    normalize,
    sentence,
    with_article,
)
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.describe import DescribePerson  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def test_parts_come_from_the_head_noun(self):
        cases = {
            "Red T-shirt": ("upper", "red t-shirt"),
            "red tshirt": ("upper", "red t-shirt"),
            "red t shirt": ("upper", "red t-shirt"),
            "red tee shirt": ("upper", "red t-shirt"),
            "red tee-shirt": ("upper", "red t-shirt"),
            "red tee": ("upper", "red t-shirt"),
            "blue jeans": ("lower", "blue jeans"),
            "short black hair": ("hair", "short black hair"),
            "black-framed glasses": ("glasses", "black-framed glasses"),
            "bald": ("hair", "bald"),
            "white sneakers": ("shoes", "white sneakers"),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(normalize(phrase), expected)

    def test_leading_verbs_and_articles_are_dropped(self):
        cases = {
            "wearing glasses": ("glasses", "glasses"),
            "has a beard": ("facial_hair", "beard"),
            "a red t-shirt": ("upper", "red t-shirt"),
            "carrying a backpack": ("bag", "backpack"),
            "is wearing a pair of black glasses": ("glasses", "black glasses"),
            "dressed in a green hoodie": ("upper", "green hoodie"),
            "has short black hair": ("hair", "short black hair"),
            "is bald": ("hair", "bald"),
            # Counted without the dropped words: 4 of 7.
            "is wearing a light blue denim shirt": ("upper", "light blue denim shirt"),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(normalize(phrase), expected)
        for phrase in (
            "wearing a light blue denim button shirt",  # still 5 words
            "wearing a",
            "has no beard",
            "wearing hair",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(normalize(phrase))

    def test_held_things_are_told_only_of_a_bag(self):
        cases = {
            "carrying a backpack": ("bag", "backpack"),
            "holding a black handbag": ("bag", "black handbag"),
            "bag on shoulder": ("bag", "bag"),
            "black backpack on her back": ("bag", "black backpack"),
            "red cap on head": ("headwear", "red cap"),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(normalize(phrase), expected)
        for phrase in (
            "carrying a black jacket",
            "holding a red cap",
            "jacket over arm",
            "black jacket over his left arm",
            "cap in hand",
            "coat held in hands",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(normalize(phrase))

    def test_details_after_with_are_dropped(self):
        self.assertEqual(
            normalize("red t-shirt with a white logo"), ("upper", "red t-shirt")
        )

    def test_hedged_negated_unknown_and_long_phrases_are_not_told(self):
        for phrase in (
            "possibly glasses",
            "no glasses",
            "not wearing a hat",
            "red or orange shirt",
            "smiling",
            "slim build",
            "light blue denim button up shirt",
            "hair",
            "hair in a bun",
            "",
            None,
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(normalize(phrase))


class ConsensusTests(unittest.TestCase):
    def test_a_feature_of_one_view_is_dropped(self):
        views = [
            ["green t-shirt", "short blond hair"],
            ["Green T-shirt", "short hair", "glasses"],
            ["green tshirt", "short blonde hair"],
        ]
        self.assertEqual(consensus(views), ["green t-shirt", "short blond hair"])

    def test_a_part_whose_colour_the_views_disagree_on_is_dropped(self):
        views = [["black pants"], ["dark grey pants"], ["blue jeans"]]
        self.assertEqual(consensus(views), [])

    def test_two_of_three_agree(self):
        views = [["black pants", "beard"], ["dark grey pants"], ["black pants"]]
        self.assertEqual(consensus(views), ["black pants"])

    def test_garment_words_of_one_kind_agree_on_the_colour(self):
        views = [
            ["red t-shirt", "blue jeans"],
            ["red shirt", "blue pants"],
            ["red top", "blue jeans"],
        ]
        # No word has a majority for the shirt: told as colour and kind.
        self.assertEqual(consensus(views), ["red shirt", "blue jeans"])

    def test_hair_without_an_agreed_colour_is_not_told(self):
        self.assertEqual(consensus([["short hair"], ["long hair"], ["hair"]]), [])
        # "hair in a bun" is "hair" without the detail: kept as text, never told.
        self.assertEqual(choose_features(consensus([["hair in a bun"]] * 3)), [])
        self.assertEqual(choose_features(["hair", "glasses"]), [("glasses", "glasses")])
        self.assertEqual(
            consensus([["short black hair"], ["black hair"], ["long black hair"]]),
            ["black hair"],
        )

    def test_failed_and_missing_views_count_against(self):
        self.assertEqual(consensus([["red t-shirt", "glasses"]], 3), [])
        self.assertEqual(consensus([["red t-shirt", "glasses"], None], 3), [])
        self.assertEqual(
            consensus([["red t-shirt", "glasses"], None, ["red t-shirt"]], 3),
            ["red t-shirt"],
        )

    def test_phrases_of_no_part_need_the_same_text(self):
        views = [["ground truth guest1"], ["ground truth guest1"]]
        self.assertEqual(consensus(views), ["ground truth guest1"])
        self.assertEqual(consensus([["smiling"], ["frowning"]]), [])

    def test_one_view_asked_is_kept_as_it_is(self):
        self.assertEqual(
            consensus([["red t-shirt", "glasses"]]), ["red t-shirt", "glasses"]
        )
        self.assertEqual(consensus([]), [])


class ChooseTests(unittest.TestCase):
    def test_one_per_part_in_reliability_order_at_most_four(self):
        appearance = [
            "white sneakers",
            "short black hair",
            "red t-shirt",
            "red shirt",
            "blue jeans",
            "glasses",
            "beard",
        ]
        self.assertEqual(
            choose_features(appearance),
            [
                ("upper", "red t-shirt"),
                ("glasses", "glasses"),
                ("lower", "blue jeans"),
                ("hair", "short black hair"),
            ],
        )
        self.assertEqual(len(choose_features(appearance, 2)), 2)

    def test_nothing_usable(self):
        self.assertEqual(choose_features(["smiling", "no hat"]), [])


class SentenceTests(unittest.TestCase):
    def test_wearing_and_having(self):
        features = [
            ("upper", "orange t-shirt"),
            ("glasses", "glasses"),
            ("lower", "blue jeans"),
            ("hair", "short black hair"),
        ]
        self.assertEqual(
            sentence("Charlie", features, "Emma"),
            "Emma, Charlie is wearing an orange t-shirt, glasses and blue jeans "
            "and has short black hair.",
        )

    def test_bald_beard_and_bag(self):
        features = [("hair", "bald"), ("facial_hair", "beard"), ("bag", "backpack")]
        self.assertEqual(
            sentence("the other guest", features),
            "The other guest has a beard and a backpack and is bald.",
        )

    def test_verbs_and_articles_of_the_vlm_are_not_repeated(self):
        features = choose_features(
            ["wearing glasses", "has a beard", "a red t-shirt", "carrying a backpack"]
        )
        self.assertEqual(
            sentence("Charlie", features, "Emma"),
            "Emma, Charlie is wearing a red t-shirt and glasses "
            "and has a beard and a backpack.",
        )
        views = [["wearing glasses"], ["wearing glasses"], ["glasses"]]
        self.assertEqual(consensus(views), ["glasses"])

    def test_only_a_sentence_without_listener_is_capitalized(self):
        features = [("glasses", "glasses")]
        self.assertEqual(
            sentence("the other guest", features, "Emma"),
            "Emma, the other guest is wearing glasses.",
        )
        self.assertEqual(
            sentence("the other guest", features), "The other guest is wearing glasses."
        )
        features = choose_features(["carrying a black jacket", "carrying a backpack"])
        self.assertEqual(
            sentence("Charlie", features, "Emma"), "Emma, Charlie has a backpack."
        )

    def test_articles_follow_the_sound(self):
        cases = {
            "uniform": "a uniform",
            "blue uniform": "a blue uniform",
            "orange t-shirt": "an orange t-shirt",
            "olive jacket": "an olive jacket",
            "unbuttoned shirt": "an unbuttoned shirt",
            "used bag": "a used bag",
            "one-piece dress": "a one-piece dress",
            "hoodie": "a hoodie",
            "helmet": "a helmet",
            "eyeglasses": "eyeglasses",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(with_article(phrase), expected)
        self.assertEqual(
            sentence("Charlie", [("upper", "uniform")]), "Charlie is wearing a uniform."
        )

    def test_no_features_no_sentence(self):
        self.assertEqual(sentence("Charlie", []), "")


class FakePeople:
    look_settle = 1.5

    def __init__(self, message=None, following=None):
        self.message = message
        self.followed = []
        self.paused = []
        self.head = following

    def following(self):
        return self.head

    def follow(self, track=None, name=""):
        self.followed.append((track, name))
        self.head = (track, name)

    def latest(self):
        return self.message

    def pause(self, seconds, cancelled):
        self.paused.append(seconds)


class FakeInteraction:
    def __init__(self, ok=True):
        self.ok = ok
        self.said = []

    def say(self, text, cancelled):
        self.said.append(text)
        return self.ok, "spoken" if self.ok else "TTS failed"


def run(params, memory, people=None, speech=None):
    people = people or FakePeople()
    speech = speech or FakeInteraction()
    runtime = SimpleNamespace(node=None, people=people, interaction=speech)
    ctx = SkillContext(
        subgoal="guest2.describe_guest1",
        params=params,
        memory=memory,
        cancelled=threading.Event(),
        log=lambda text: None,
    )
    return DescribePerson(runtime).execute(ctx), people, speech


PARAMS = {"subject": "guest1", "listener": "guest2"}


class DescribePersonTests(unittest.TestCase):
    def test_tells_the_listener_while_looking_at_them(self):
        memory = {
            "guest1.appearance": ["red t-shirt", "blue jeans", "smiling"],
            "guest1.name": "Charlie",
            "guest2.name": "Emma",
            "guest2.track": 4,
        }
        visible = SimpleNamespace(id=4, name="guest2", visible=True)
        result, people, speech = run(
            PARAMS, memory, FakePeople(SimpleNamespace(people=[visible]))
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            speech.said, ["Emma, Charlie is wearing a red t-shirt and blue jeans."]
        )
        self.assertEqual(people.followed, [(4, "guest2")])
        self.assertEqual(people.paused, [1.5])  # the head turns first
        self.assertEqual(result.outputs["guest1.told"], ["red t-shirt", "blue jeans"])

    def test_cancelled_while_the_head_turns_says_nothing(self):
        visible = SimpleNamespace(id=4, name="guest2", visible=True)
        people = FakePeople(SimpleNamespace(people=[visible]))
        people.pause = lambda seconds, cancelled: cancelled.set()
        result, _, speech = run(
            PARAMS, {"guest1.appearance": ["glasses"], "guest2.track": 4}, people
        )
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Cancelled")
        self.assertEqual(speech.said, [])

    def test_no_wait_when_the_head_already_follows_the_listener(self):
        visible = SimpleNamespace(id=6, name="guest2", visible=True)
        people = FakePeople(SimpleNamespace(people=[visible]), following=(6, "guest2"))
        result, people, speech = run(
            PARAMS, {"guest1.appearance": ["glasses"], "guest2.track": 3}, people
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(people.paused, [])
        self.assertEqual(len(speech.said), 1)

    def test_without_names_it_says_the_other_guest(self):
        result, _, speech = run(PARAMS, {"guest1.appearance": ["glasses"]})
        self.assertTrue(result.success)
        self.assertEqual(speech.said, ["The other guest is wearing glasses."])

    def test_nothing_certain_is_not_said(self):
        result, people, speech = run(PARAMS, {"guest1.appearance": ["maybe a hat"]})
        self.assertTrue(result.success)
        self.assertEqual(speech.said, [])
        self.assertEqual(result.outputs["guest1.told"], [])

    def test_missing_appearance_fails(self):
        result, _, speech = run(PARAMS, {})
        self.assertFalse(result.success)
        self.assertIn("guest1.appearance", result.message)
        self.assertEqual(speech.said, [])

    def test_speech_failure_fails(self):
        result, _, _ = run(
            PARAMS, {"guest1.appearance": ["glasses"]}, speech=FakeInteraction(ok=False)
        )
        self.assertFalse(result.success)
        self.assertIn("TTS failed", result.message)

    def test_params_are_validated(self):
        spec = SimpleNamespace(locations={})
        DescribePerson.validate({**PARAMS, "max_features": 3}, spec)
        for params in (
            {},
            {"subject": "guest1"},
            {"subject": "guest1", "listener": "guest1"},
            {**PARAMS, "max_features": 0},
            {**PARAMS, "max_features": True},
            {**PARAMS, "extra": 1},
        ):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    DescribePerson.validate(params, spec)


if __name__ == "__main__":
    unittest.main()
