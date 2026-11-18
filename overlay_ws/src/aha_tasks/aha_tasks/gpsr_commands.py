"""GPSR task flow: hear the commands, plan them, and report what was done.

`listen_command` asks the operator for a command through the ASR
(/aha/perception/audio/listen, with the arena's words as hotwords), repeats
what it heard on the screen and by speech, and asks for a yes / no. A "no", an
empty or too short transcript and silence make it ask again (a rephrasing
request); after the referee's attempts it asks for a custom operator.
`plan_commands` turns the heard texts into one plan (aha_tasks.gpsr_planner),
shows it on the screen as a picture and says it. `report_results` says per
command what was done, from the plan, the task status and the answers in
memory.
"""

import functools
import re
import threading
import time

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.gpsr_planner import (
    CommandPlan,
    Planner,
    command_summary,
    interleaved,
    plan_speech,
    plan_travel,
    render_plan,
    report_sentences,
    sequential,
    step_outcomes,
)
from aha_tasks.gpsr_vocabulary import load_vocabulary, names_of
from aha_tasks.interaction import NO_ASR, Cancelled, Interaction
from aha_tasks.plan import COMMANDS, PLAN, validate_plan
from aha_tasks.task_status import TaskStatusSkill, planning_skipped, unheard
from aha_tasks.skills import LOCATION

ORDINALS = {1: "first", 2: "second", 3: "third"}
# Words of an answer to "Is that correct?". A bare "not" or "n't" is no answer
# ("I didn't hear you", "Sure, why not"), and asking back ("What?",
# "Pardon?") is neither an answer nor a command.
YES = {
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
}
NO = {"no", "nope", "nah", "wrong", "incorrect", "negative", "never"}
NO_PHRASE = re.compile(
    r"^not\b(?! sure\b)|\b(?:that's|that is|it's|it is) not\b(?! sure\b)"
    r"|\b(?:isn't|wasn't)\b"
)
FILLERS = re.compile(
    r"^(?:i said|i say|it is|it's|its|it was|the command is|i mean)\s+"
)
# Words before a restated command: "No, no, sorry, I said ...".
OPENERS = {"sorry", "well", "um", "uh", "oh", "hmm", "wait", "actually"}
LEADING = YES | NO | OPENERS | {"quite", "really"}
MIN_WORDS = 3  # a command has at least this many words
PLANNERS = ("auto", "llm", "grammar")
# Interaction.listen's reasons for silence (the ASR's "no transcript within").
SILENCE = ("ASR: no transcript", "ASR: nothing was said")
ASR_RETRY = 1.0  # s between tries after another ASR error


def words_of(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower().replace("’", "'"))


def yes_no(text):
    """("yes" | "no" | None, the rest) of an answer to "Is that correct?".

    The rest is what follows a leading yes / no word, or the whole answer
    without one, e.g. a correction "No, I said bring me ...".
    """
    words = words_of(text)
    # "Sorry, no, I said ...", "Oh no, ...", "Um, yes": judge from the answer.
    while words and words[0] in OPENERS:
        words = words[1:]
    if not words:
        return None, ""
    joined = " ".join(words)
    if words[0] in YES and words[1:2] == ["not"]:
        return "no", ""  # "Absolutely not", "Certainly not"
    if words[0] in YES:
        return "yes", " ".join(words[1:])
    denial = NO_PHRASE.search(joined)
    if words[0] in NO or denial:
        # "No, no, sorry, bring me ...", "That's not right, I said bring me ...":
        # the command after the denial and the leading words.
        rest = words[1:] if words[0] in NO else joined[denial.end() :].split()
        while rest and rest[0] in LEADING:
            rest = rest[1:]
        return "no", " ".join(rest)
    if len(words) <= 3 and any(w in NO for w in words):
        return "no", ""
    # A yes word right after "not" ("I'm not sure") is no yes.
    negated = {i + 1 for i, w in enumerate(words) if w == "not" or w.endswith("n't")}
    if len(words) <= 3 and any(
        w in YES for i, w in enumerate(words) if i not in negated
    ):
        return "yes", ""
    return None, joined


def affirms(text):
    """A yes not at the start of a longer answer ("That's right, thank you"):
    a yes word and no no word, "not" or "n't" anywhere ("I don't think that's
    right" is no yes)."""
    words = words_of(text)
    if any(w in NO or w == "not" or w.endswith("n't") for w in words):
        return False
    return any(w in YES for w in words)


def restated(rest, vocabulary):
    """The command in an answer such as 'no, I said bring me ...', or None.

    It needs more than MIN_WORDS words and a word of the arena's vocabulary
    (a name, a place, an object or category), as every generator command
    has: "I didn't hear it" or "Could you say it again?" is no command.
    """
    words = FILLERS.sub("", rest.strip()).split()
    while words and words[-1] in YES:  # "bring me ..., right?"
        words.pop()
    rest = " ".join(words)
    if len(words) <= MIN_WORDS:
        return None
    for word in vocabulary.hotwords():
        if re.search(rf"\b{re.escape(word.lower())}\b", rest):
            return rest
    return None


def clean(text):
    return " ".join(str(text).split()).strip(" .,?!")


@functools.lru_cache(maxsize=8)
def _vocabulary(name, locations, names):
    return load_vocabulary(name, dict(locations), names)


def vocabulary_of(params, spec):
    """The vocabulary named by params["vocabulary"] (default: the task's name).

    People's names are the task's initial_state `gpsr.names`.
    """
    name = params.get("vocabulary", spec.task)
    if not isinstance(name, str) or not name:
        raise ValueError(f"vocabulary must be a name or JSON path: {name!r}")
    locations = tuple((k, tuple(v)) for k, v in spec.locations.items())
    return _vocabulary(name, locations, names_of(spec.initial_state))


def number(params, key, default, low=0.0):
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < low:
        raise ValueError(f"{key} must be a number >= {low}: {value!r}")
    return value


def integer(params, key, default, low=0):
    value = params.get(key, default)
    if type(value) is not int or value < low:
        raise ValueError(f"{key} must be an integer >= {low}: {value!r}")
    return value


class Lost(RuntimeError):
    """The ASR is not served, or it keeps failing."""


def hear(interaction, hotwords, timeout, cancelled):
    """(transcript, "") or (None, why) of silence, by Interaction.listen.

    Another ASR error (a goal rejected while another client listens, no
    answer from the server) is no answer, not silence: listening is tried
    again until `timeout` has passed. Raises Cancelled, or Lost when there is
    no ASR or it still fails then.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = max(1.0, deadline - time.monotonic())
        text, why = interaction.listen(hotwords, cancelled, timeout=remaining)
        if why == "Cancelled" or cancelled.is_set():
            raise Cancelled
        if text is not None or why.startswith(SILENCE):
            return text, why
        if why.startswith(NO_ASR) or time.monotonic() + ASR_RETRY >= deadline:
            raise Lost(why)
        if cancelled.wait(ASR_RETRY):
            raise Cancelled


class ListenCommand(Skill):
    """Hear one command from the operator and confirm it.

    params: {"command": 1-3, "timeout": 15.0 (s to wait for the command),
    "confirm_timeout": 8.0, "attempts": 3 (the referee's tries: the command
    and 2 rephrasings), "custom_operator": 3 (tries of a custom operator
    afterwards, 0: none), "vocabulary": "<config/vocabulary name>"}.
    Outputs `cmd<N>.text`.

    Rulebook §5.3: the referee has three re-phrased commands; "After the
    referee attempts the three re-phrased commands, the custom operator can be
    requested", who "only has three attempts". Score: "Request a rephrasing
    6×–30" (2 per command), "Using a custom operator 3×–20".
    """

    KEYS = {"command", "timeout", "confirm_timeout", "attempts", "custom_operator"}

    @classmethod
    def validate(cls, params, spec):
        command = params.get("command")
        unknown = set(params) - cls.KEYS - {"vocabulary"}
        if unknown or type(command) is not int or command not in COMMANDS:
            raise ValueError(f"listen_command needs a command of {COMMANDS}: {params}")
        number(params, "timeout", 15.0, low=1.0)
        number(params, "confirm_timeout", 8.0, low=1.0)
        integer(params, "attempts", 3, low=1)
        integer(params, "custom_operator", 3)
        vocabulary_of(params, spec)

    def ready(self):
        interaction = Interaction.of(self.runtime)
        interaction.find_tts()
        interaction.find_asr()

    def execute(self, ctx):
        params = ctx.params
        n = params["command"]
        ordinal = ORDINALS[n]
        interaction = Interaction.of(self.runtime)
        vocabulary = vocabulary_of(params, self.runtime.spec)
        hotwords = vocabulary.hotwords()
        timeout = float(params.get("timeout", 15.0))
        attempts = params.get("attempts", 3)
        tries = attempts + params.get("custom_operator", 3)
        prompt = f"Please tell me the {ordinal} command."
        try:
            # Every referee's attempt after the first is a rephrasing request.
            for attempt in range(tries):
                if attempt == attempts:
                    prompt = (
                        "I am sorry. Could a custom operator please tell me the "
                        f"{ordinal} command?"
                    )
                self._say(interaction, prompt, ctx)
                text, why = hear(interaction, hotwords, timeout, ctx.cancelled)
                text = clean(text or "")
                if not text:
                    ctx.log(f"Nothing heard: {why}")
                    prompt = (
                        f"I did not hear you. Please tell me the {ordinal} command."
                    )
                    continue
                if len(text.split()) < MIN_WORDS:
                    ctx.log(f"Too short for a command: {text!r}")
                    prompt = (
                        f"Sorry, I did not understand. Please tell me the {ordinal} "
                        "command again."
                    )
                    continue
                confirmed = self._confirm(interaction, text, vocabulary, ctx)
                if confirmed:
                    asked = f"{min(attempt, attempts - 1)} rephrasing request(s)"
                    if attempt >= attempts:
                        asked += f", custom operator try {attempt - attempts + 1}"
                    return SkillResult(
                        True,
                        f"Command {n}: {confirmed!r} ({asked})",
                        {f"cmd{n}.text": confirmed},
                    )
                prompt = f"Sorry. Please tell me the {ordinal} command again."
        except Cancelled:
            return SkillResult(False, "Cancelled")
        except Lost as error:
            return SkillResult(False, str(error))
        return SkillResult(
            False, f"Did not understand the {ordinal} command in {tries} tries"
        )

    def _say(self, interaction, text, ctx):
        ok, message = interaction.say(text, ctx.cancelled)
        if ctx.cancelled.is_set():
            raise Cancelled
        if not ok:
            ctx.log(f"Could not say {text!r}: {message}")

    def _confirm(self, interaction, text, vocabulary, ctx):
        """The command once the operator says yes, or None on a no."""
        timeout = float(ctx.params.get("confirm_timeout", 8.0))
        silences = 0  # hear raises on other ASR errors
        corrected = False  # the text is a restatement from an answer
        for _ in range(4):
            self._say(
                interaction,
                f"I heard: {text}. Is that correct? Please say yes or no.",
                ctx,
            )
            answer, why = hear(interaction, ["yes", "no"], timeout, ctx.cancelled)
            verdict, rest = yes_no(answer or "")
            ctx.log(f"Confirmation of {text!r}: {answer!r} -> {verdict}")
            if verdict == "yes":
                return text
            correction = restated(rest, vocabulary)
            if correction:  # "no, I said ..." or the command once more
                text, corrected = clean(correction), True
                continue
            if verdict == "no":
                return None
            if affirms(rest):  # "That's right, thank you"; no command restated
                return text
            if corrected:
                # A restated command (heard once) is taken only on a yes.
                return None
            if answer:
                # Heard, but neither yes nor no ("I don't think so"): ask
                # again rather than take a command the operator may deny.
                continue
            silences += 1
            if silences >= 2:
                ctx.log(f"No answer to the confirmation ({why}); taking {text!r}")
                return text
        return None  # no yes in the tries (or the last answer restated the command)


def command_infos(results, count, vocabulary):
    """`plan.commands`: every command 1..count, a not heard one included, so
    that report_results says something about each."""
    planned = {r.number: r for r in results}
    return [
        planned.get(n, CommandPlan(n, "", error="not heard")).info(vocabulary)
        for n in range(1, count + 1)
    ]


class PlanCommands(TaskStatusSkill):
    """Plan the heard commands, show the plan on the screen and say it.

    params: {"commands": ["cmd1.text", ...] (memory keys, in command order),
    "timeout": 60.0 (s for planning), "attempts": 3 (LLM tries per command),
    "planner": "auto" | "llm" | "grammar", "thinking": false,
    "vocabulary": "<config/vocabulary name>"}. The LLM is the runner's ROS
    parameters llm_url / llm_model (aha_tasks.llm). Outputs `plan` and
    `plan.commands` (per command: text, planner, summary, error).
    """

    KEYS = {"commands", "timeout", "attempts", "planner", "thinking", "vocabulary"}

    @classmethod
    def validate(cls, params, spec):
        commands = params.get("commands")
        if set(params) - cls.KEYS or not isinstance(commands, list):
            raise ValueError(f"plan_commands needs commands (memory keys): {params}")
        if not 1 <= len(commands) <= len(COMMANDS) or not all(
            isinstance(k, str) and k for k in commands
        ):
            raise ValueError(f"commands must be 1 to 3 memory keys: {commands!r}")
        number(params, "timeout", 60.0, low=1.0)
        integer(params, "attempts", 3, low=1)
        if params.get("planner", "auto") not in PLANNERS:
            raise ValueError(f"planner must be one of {PLANNERS}: {params}")
        if not isinstance(params.get("thinking", False), bool):
            raise ValueError(f"thinking must be true or false: {params}")
        vocabulary_of(params, spec)

    def ready(self):
        Interaction.of(self.runtime).find_tts()
        super().ready()

    def planner(self, params):
        from aha_tasks.llm import Chat, parameters

        vocabulary = vocabulary_of(params, self.runtime.spec)
        url, model = parameters(self.runtime.node)
        return Planner(
            vocabulary,
            self.runtime.spec,
            Chat(url, model),
            mode=params.get("planner", "auto"),
            attempts=params.get("attempts", 3),
            thinking=params.get("thinking", False),
        )

    def execute(self, ctx):
        params, spec = ctx.params, self.runtime.spec
        vocabulary = vocabulary_of(params, spec)
        skipped = unheard(params["commands"], spec, self.latest_status(ctx.subgoal))
        for key in sorted(skipped):
            ctx.log(f"{key} was not heard (its listening was skipped): not planned")
        texts = {
            n: ctx.memory.get(key)
            for n, key in enumerate(params["commands"], 1)
            if isinstance(ctx.memory.get(key), str) and key not in skipped
        }
        start = ctx.memory.get(LOCATION)
        if not isinstance(start, str) or start not in spec.locations:
            start = vocabulary.operator
        planner = self.planner(params)
        done, box = threading.Event(), {}

        def work():
            try:
                box["result"] = planner.plan(
                    texts, float(params.get("timeout", 60.0)), ctx.cancelled, start
                )
            except Exception as error:  # reported below
                box["error"] = error
            done.set()

        started = time.monotonic()
        threading.Thread(target=work, daemon=True).start()
        while not done.wait(0.1):
            if ctx.cancelled.is_set():
                return SkillResult(False, "Cancelled")
        if "error" in box:
            return SkillResult(False, f"Planning failed: {box['error']}")
        plan, results = box["result"]
        for result in results:
            tries = ", ".join(
                f"{seconds:.1f} s: {'; '.join(errors) or 'ok'}"
                if seconds is not None
                else "; ".join(errors)
                for seconds, errors in result.attempts
            )
            ctx.log(
                f"Command {result.number} ({result.planner or 'failed'}): "
                f"{result.text!r}"
                + (f" [{tries}]" if tries else "")
                + (f" {result.error}" if result.error and not result.steps else "")
            )
        interaction = Interaction.of(self.runtime)
        count = len(params["commands"])
        understood = [r.number for r in results if r.steps]
        dropped = [r.number for r in results if r.dropped]
        missing = [
            n for n in range(1, count + 1) if n not in understood and n not in dropped
        ]
        if not plan:
            interaction.say("I am sorry, I could not plan the commands.", ctx.cancelled)
            return SkillResult(False, "No command could be planned")
        try:
            validate_plan(plan, spec)
        except ValueError as error:
            return SkillResult(False, f"Invalid plan: {error}")
        commands = {r.number: r.steps for r in results if r.steps}
        ends = (start, vocabulary.operator)
        travel = plan_travel(plan, spec, *ends)
        one_by_one = plan_travel(sequential(commands), spec, *ends)
        speech = [f"I understood {_numbers(understood)}."]
        if missing:
            speech.append(f"I could not understand {_numbers(missing)}.")
        if dropped:
            speech.append(
                f"I understood {_numbers(dropped)}, but it does not fit my plan."
            )
        if interleaved(plan):
            speech.append("I interleave the commands to save travel.")
        speech.append(plan_speech(plan, vocabulary))
        image = render_plan(plan, vocabulary)
        ok, message = interaction.show(image, "png", " ".join(speech), ctx.cancelled)
        if ctx.cancelled.is_set():
            return SkillResult(False, "Cancelled")
        if not ok:
            ctx.log(f"Could not say the plan: {message}")
        return SkillResult(
            True,
            f"{len(plan)} steps for {_numbers(understood)} in "
            f"{time.monotonic() - started:.1f} s; travel {travel:.1f} m "
            f"({one_by_one:.1f} m one by one)",
            {
                PLAN: plan,
                f"{PLAN}.commands": command_infos(results, count, vocabulary),
            },
        )


def _numbers(numbers):
    if len(numbers) == 1:
        return f"command {numbers[0]}"
    words = ", ".join(str(n) for n in numbers[:-1])
    return f"commands {words} and {numbers[-1]}"


class ReportResults(TaskStatusSkill):
    """Say per command what was done.

    Uses the plan and `plan.commands` in memory, the outcome of each plan slot
    from the task status (/aha/tasks/status), and the answers of count,
    count_people and describe (`<output>`, `<output>.answer`).
    params (optional): {"vocabulary": "<config/vocabulary name>"}.
    """

    @classmethod
    def validate(cls, params, spec):
        if set(params) - {"vocabulary"}:
            raise ValueError(f"report_results takes only vocabulary: {params}")
        vocabulary_of(params, spec)

    def ready(self):
        Interaction.of(self.runtime).find_tts()
        super().ready()

    def execute(self, ctx):
        status = self.latest_status(ctx.subgoal)
        if planning_skipped(self.runtime.spec, status):
            # The plan in memory is the task definition's mock, not the
            # operator's commands: say none were planned.
            return self._report(
                ctx, "I could not plan the commands, so I did not do them."
            )
        plan = ctx.memory.get(PLAN)
        if not isinstance(plan, list) or not plan:
            return SkillResult(False, "No plan in memory")
        commands = ctx.memory.get(f"{PLAN}.commands")
        if not isinstance(commands, list):  # a plan not made by plan_commands
            vocabulary = vocabulary_of(ctx.params, self.runtime.spec)
            numbers = sorted({step["command"] for step in plan})
            commands = [
                {
                    "command": n,
                    "summary": command_summary(
                        [s for s in plan if s["command"] == n], vocabulary
                    ),
                }
                for n in numbers
            ]
        outcomes = step_outcomes(status, plan)
        sentences = report_sentences(plan, commands, outcomes, ctx.memory)
        return self._report(ctx, " ".join(sentences))

    def _report(self, ctx, summary):
        interaction = Interaction.of(self.runtime)
        interaction.hide()
        ok, message = interaction.say("Here is my report. " + summary, ctx.cancelled)
        if ctx.cancelled.is_set():
            return SkillResult(False, "Cancelled")
        if not ok:
            ctx.log(f"Could not say the report: {message}")
        return SkillResult(True, summary)
