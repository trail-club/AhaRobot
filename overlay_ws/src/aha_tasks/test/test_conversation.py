"""GPSR conversation text: names, replies, questions, sentences and the dialogue."""

import datetime
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.conversation import (  # noqa: E402
    KNOWN_QUESTIONS,
    NAMES,
    Dialogue,
    answer,
    count_sentence,
    extract_name,
    facts_of,
    find_name,
    is_named,
    local_now,
    names_of,
    parse_reply,
    person_info_sentence,
    question_topic,
    soundex,
    value_sentence,
)
from aha_tasks.interaction import NO_ASR  # noqa: E402
from aha_tasks.person_description import Description  # noqa: E402

FACTS = {
    "gpsr.robot_name": "AhaRobot",
    "gpsr.team_name": "TRAIL",
    "gpsr.team_country": "Japan",
    "gpsr.team_affiliation": "the University of Tokyo",
}
MONDAY = datetime.datetime(2026, 11, 16, 15, 5)


class FakeInteraction:
    """Says nothing aloud; hears `heard` (text or None with a reason) in turn."""

    def __init__(self, heard=(), touched=(None, "timeout")):
        self.heard = list(heard)
        self.touched = touched
        self.calls = []

    def say(self, text, cancelled):
        self.calls.append(("say", text))
        return True, "said"

    def listen(self, hotwords, cancelled, timeout=10.0):
        self.calls.append(("listen", tuple(hotwords)))
        if not self.heard:
            return None, "ASR: nothing was said"
        item = self.heard.pop(0)
        return (item, "") if isinstance(item, str) or item is None else item

    def ask(self, prompt, choices, cancelled, timeout=0.0):
        self.calls.append(("ask", prompt, tuple(choices)))
        return self.touched

    def kinds(self):
        return [call[0] for call in self.calls]


def dialogue(interaction, logs=None):
    return Dialogue(
        interaction,
        NAMES,
        threading.Event(),
        (logs if logs is not None else []).append,
        touch=True,
    )


class NameTests(unittest.TestCase):
    def test_exact_names_and_the_one_after_a_cue_first(self):
        self.assertEqual(find_name("Charlie", NAMES), "Charlie")
        self.assertEqual(find_name("Yes, I am charlie.", NAMES), "Charlie")
        self.assertEqual(find_name("I'm Jane, not Charlie.", NAMES), "Jane")
        self.assertIsNone(find_name("Yes.", NAMES))

    def test_misheard_names_by_spelling_and_sound(self):
        for heard, name in {
            "My name is Jewels.": "Jules",
            "Axle": "Axel",
            "I'm Adele": "Adel",
            "Robyn": "Robin",
            "It's Charly": "Charlie",
            "Simon.": "Simone",
        }.items():
            with self.subTest(heard):
                self.assertEqual(find_name(heard, NAMES), name)
        self.assertEqual(soundex("Jewels"), soundex("Jules"))

    def test_other_words_are_not_names(self):
        for heard in (
            "My name is John.",
            "I think the person over there by the window is waiting",
            "Hmm, what?",
        ):
            with self.subTest(heard):
                self.assertIsNone(find_name(heard, NAMES))

    def test_a_name_outside_the_list(self):
        self.assertEqual(extract_name("My name is Jewels", NAMES), "Jules")
        # Only the task's names: other words after a cue are no name.
        for heard in (
            "Hi, I'm Kenji.",
            "Kenji",
            "I'm fine",
            "Call me later",
            "Jane is over there",
            "Ask Charlie, not me",
            "Sorry, what did you say?",
            "What?",
            "Pardon?",
            "Huh?",
        ):
            with self.subTest(heard):
                self.assertIsNone(extract_name(heard, NAMES))

    def test_names_of_the_task(self):
        self.assertEqual(names_of({}), list(NAMES))
        self.assertEqual(names_of({"gpsr.names": [" Kenji "]}), ["Kenji"])
        with self.assertRaises(ValueError):
            names_of({"gpsr.names": ["Kenji", 3]})


class ReplyTests(unittest.TestCase):
    def test_replies_to_are_you_charlie(self):
        cases = {
            "Yes.": True,
            "Yeah, that's me!": True,
            "Yes, I am Charlie.": True,
            "I am.": True,
            "Charlie.": True,
            "Charly": True,
            "No.": False,
            "Nope": False,
            "No, I'm Jane.": False,
            "I'm not Charlie.": False,
            "I am Jane": False,
            "Jane": False,
            "Hmm, what?": None,
            "Could you repeat that": None,
            "That's not me.": False,
            "Not me": False,
            "Not really": False,
            "Sure, why not": True,
            "Sorry, I didn't hear you": None,
            "I can't hear you": None,
            "I don't understand": None,
            "Excuse me?": None,
            "I'm not sure": None,
            "Not sure": None,
            "I am not able to hear you": None,
            "It's not clear": None,
            "I'm not.": False,
            "No, I'm not Charlie": False,
            "Charlie is over there": None,
            "This is Charlie's seat": None,
            "I'll call Charlie for you": None,
            "Yes, Charlie": True,
            "Yes, my name is Charly, nice to meet you": True,
            "Absolutely not": False,
            "Certainly not.": False,
            "She's right there": None,
            "He is right over there": None,
            "Yes, of course": True,
            "Charlie is right there": None,
            "Sure, Charlie is over there": None,
            "Okay, Charlie is sitting over there": None,
        }
        for heard, expected in cases.items():
            with self.subTest(heard):
                reply = parse_reply(heard, NAMES)
                self.assertIs(is_named(reply, "Charlie"), expected)


class QuestionTests(unittest.TestCase):
    def test_generator_questions_and_rephrasings(self):
        cases = {
            "something about yourself": "about_yourself",
            "introduce yourself": "about_yourself",
            "tell me who you are": "about_yourself",
            "the time": "time",
            "what time is it": "time",
            "what day is today": "today",
            "today's date": "today",
            "what day is tomorrow": "tomorrow",
            "the date of tomorrow": "tomorrow",
            "your teams name": "team_name",
            "the name of your team": "team_name",
            "your team's country": "team_country",
            "where your team comes from": "team_country",
            "your teams affiliation": "team_affiliation",
            "which university your team belongs to": "team_affiliation",
            "the day of the week": "day_of_week",
            "the day of the month": "day_of_month",
            "team_name": "team_name",
        }
        for question, topic in cases.items():
            with self.subTest(question):
                self.assertEqual(question_topic(question), topic)
        for question in KNOWN_QUESTIONS:
            self.assertIsNotNone(question_topic(question))
        self.assertIsNone(question_topic("the capital of France"))

    def test_answers(self):
        facts = facts_of(FACTS)
        cases = {
            "time": "It is 3:05 PM.",
            "today": "Today is Monday, November 16th, 2026.",
            "tomorrow": "Tomorrow is Tuesday, November 17th, 2026.",
            "day_of_week": "Today is Monday.",
            "day_of_month": "Today is the 16th of November.",
            "team_name": "My team's name is TRAIL.",
            "team_country": "My team is from Japan.",
            "team_affiliation": "My team's affiliation is the University of Tokyo.",
            "about_yourself": "My name is AhaRobot. I am a service robot of team "
            "TRAIL from the University of Tokyo in Japan.",
        }
        for topic, sentence in cases.items():
            with self.subTest(topic):
                self.assertEqual(answer(topic, facts, MONDAY), sentence)
        self.assertEqual(
            answer("time", facts, MONDAY.replace(hour=0, minute=7)), "It is 12:07 AM."
        )
        with self.assertRaises(ValueError):
            answer("team_name", facts_of({}), MONDAY)

    def test_time_zone(self):
        now = local_now({"gpsr.utc_offset": 9})
        self.assertEqual(now.utcoffset(), datetime.timedelta(hours=9))
        self.assertIsNotNone(local_now({}).utcoffset())


class SentenceTests(unittest.TestCase):
    def test_person_info(self):
        self.assertEqual(
            person_info_sentence("name", "Jane"), "The person's name is Jane."
        )
        self.assertEqual(
            person_info_sentence("pose", "lying"), "The person is lying down."
        )
        self.assertEqual(
            person_info_sentence("gesture", "raising_left_arm"),
            "The person is raising their left arm.",
        )
        self.assertEqual(
            person_info_sentence("gesture", "none"),
            "The person is not making any gesture.",
        )
        with self.assertRaises(ValueError):
            person_info_sentence("gesture", "")

    def test_values_in_memory(self):
        self.assertEqual(value_sentence("cmd2.count", 3), "The answer is 3.")
        self.assertEqual(
            value_sentence("person.pose", "sitting"), "The person is sitting."
        )
        self.assertEqual(
            value_sentence("cmd1.object", {"label": "orange_juice"}),
            "The answer is orange juice.",
        )
        self.assertEqual(
            value_sentence("x", ["cola", "milk", "tea"]),
            "The answer is cola, milk and tea.",
        )
        for empty in (None, "", []):
            with self.assertRaises(ValueError):
                value_sentence("x", empty)

    def test_counts_of_people(self):
        waving = Description(gesture="waving")
        self.assertEqual(
            count_sentence(2, waving, "office"),
            "There are 2 waving people in the office.",
        )
        self.assertEqual(
            count_sentence(1, Description(posture="sitting"), "living_room"),
            "There is one sitting person in the living room.",
        )
        self.assertEqual(
            count_sentence(0, Description(clothing=("orange", "t shirt")), "desk"),
            "There are no people wearing orange t shirts at the desk.",
        )
        self.assertEqual(
            count_sentence(1, Description(clothing=("orange", "blouse"))),
            "There is one person wearing an orange blouse.",
        )
        self.assertEqual(
            count_sentence(3, Description(gesture="pointing_left")),
            "There are 3 people pointing to the left.",
        )
        self.assertEqual(count_sentence(4, Description()), "There are 4 people.")


class DialogueTests(unittest.TestCase):
    def test_yes(self):
        interaction = FakeInteraction(["Yes, I am."])
        self.assertTrue(dialogue(interaction).confirm("Charlie"))
        self.assertEqual(
            interaction.calls,
            [
                ("say", "Excuse me, are you Charlie?"),
                ("listen", NAMES),
                ("say", "Nice to meet you, Charlie."),
            ],
        )

    def test_the_name_goes_to_the_asr_as_a_hotword(self):
        interaction = FakeInteraction(["Yes"])
        self.assertTrue(dialogue(interaction).confirm("Kenji"))
        self.assertEqual(interaction.calls[1], ("listen", (*NAMES, "Kenji")))

    def test_another_name_is_no(self):
        interaction = FakeInteraction(["My name is Jane."])
        self.assertIs(dialogue(interaction).confirm("Charlie"), False)
        self.assertEqual(interaction.calls[-1], ("say", "Thank you."))

    def test_an_unclear_answer_is_asked_again(self):
        interaction = FakeInteraction(["What?", "Yes"])
        logs = []
        self.assertTrue(dialogue(interaction, logs).confirm("Charlie"))
        self.assertEqual(interaction.kinds().count("listen"), 2)
        self.assertIn("unclear", logs[0])

    def test_the_screen_when_the_voice_fails(self):
        interaction = FakeInteraction([None, None], touched=(1, "No"))
        self.assertIs(dialogue(interaction).confirm("Charlie"), False)
        self.assertIn(("ask", "Are you Charlie?", ("Yes", "No")), interaction.calls)
        interaction = FakeInteraction([None, None])
        self.assertIsNone(dialogue(interaction).confirm("Charlie"))

    def test_without_an_asr_only_the_screen(self):
        interaction = FakeInteraction(
            [(None, f"{NO_ASR}: not served")], touched=(0, "Yes")
        )
        talk = dialogue(interaction)
        self.assertTrue(talk.confirm("Charlie"))
        self.assertFalse(talk.voice)
        self.assertEqual(interaction.kinds().count("listen"), 1)
        # The next person: no question by voice.
        self.assertTrue(talk.confirm("Jane"))
        self.assertEqual(interaction.kinds().count("listen"), 1)
        self.assertIsNone(talk.ask_name())

    def test_nobody_can_be_asked_without_asr_and_screen(self):
        interaction = FakeInteraction(
            [(None, f"{NO_ASR}: not served")], touched=(None, "No screen: gone")
        )
        talk = dialogue(interaction)
        self.assertIsNone(talk.confirm("Charlie"))
        self.assertFalse(talk.voice or talk.screen)

    def test_nothing_more_is_said_once_cancelled(self):
        for question in ("confirm", "ask_name"):
            with self.subTest(question):
                cancelled = threading.Event()
                interaction = FakeInteraction(touched=(0, "Yes"))

                def listen(hotwords, _, timeout=10.0):
                    # The skill is aborted, or its deadline passes, meanwhile.
                    interaction.calls.append(("listen", tuple(hotwords)))
                    cancelled.set()
                    return None, "Cancelled"

                interaction.listen = listen
                talk = Dialogue(interaction, NAMES, cancelled, [].append, touch=True)
                if question == "confirm":
                    self.assertIsNone(talk.confirm("Charlie"))
                else:
                    self.assertIsNone(talk.ask_name())
                # No asking again, no screen and no thanks.
                self.assertEqual(interaction.kinds(), ["say", "listen"])

    def test_ask_name(self):
        interaction = FakeInteraction(["Uh, sorry?", "My name is Jewels."])
        self.assertEqual(dialogue(interaction).ask_name(), "Jules")
        self.assertEqual(interaction.calls[0], ("say", "Excuse me, what is your name?"))
        self.assertEqual(interaction.calls[-1], ("say", "Nice to meet you, Jules."))
        self.assertIsNone(dialogue(FakeInteraction()).ask_name())


if __name__ == "__main__":
    unittest.main()
