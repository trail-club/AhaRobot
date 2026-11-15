"""describe_person: tell one guest what another looks like (HIC).

The features are `<subject>.appearance` from memorize_person; which of them
are told, and how, is aha_tasks.appearance. The robot looks at the listener
while it speaks (rulebook: look at the person it is talking to), through
aha_tasks.interaction (caption, and speech when a TTS runs).
"""

from aha_tasks.appearance import MAX_FEATURES, choose_features, sentence
from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.interaction import Interaction
from aha_tasks.people import People, check_keys, check_name


class DescribePerson(Skill):
    """Tell `listener` what `subject` looks like, looking at the listener.

    params: {"subject": "guest1", "listener": "guest2", "max_features": 4}.
    Uses `<subject>.appearance`, the names `<role>.name` if known, and
    `<listener>.track` (else the name) to look at the listener. Outputs
    `<subject>.told`, the phrases told.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, {"subject", "listener"}, {"max_features"})
        check_name(params, "subject")
        check_name(params, "listener")
        if params["subject"] == params["listener"]:
            raise ValueError(f"subject and listener must differ: {params}")
        limit = params.get("max_features", MAX_FEATURES)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"max_features must be a positive integer: {params}")

    def ready(self):
        Interaction.of(self.runtime).find_tts()  # before the first sentence

    def execute(self, ctx):
        subject, listener = ctx.params["subject"], ctx.params["listener"]
        appearance = ctx.memory.get(f"{subject}.appearance")
        if not isinstance(appearance, list):
            return SkillResult(False, f"No {subject}.appearance; memorize {subject}")
        features = choose_features(
            appearance, ctx.params.get("max_features", MAX_FEATURES)
        )
        told = {f"{subject}.told": [phrase for _, phrase in features]}
        text = sentence(
            ctx.memory.get(f"{subject}.name") or "the other guest",
            features,
            ctx.memory.get(f"{listener}.name") or "",
        )
        if not text:
            return SkillResult(
                True, f"Nothing certain to tell of {subject}: {appearance}", told
            )
        self.look_at(ctx, listener)
        ok, message = Interaction.of(self.runtime).say(text, ctx.cancelled)
        if ctx.cancelled.is_set():
            return SkillResult(False, "Cancelled")
        return SkillResult(ok, f'"{text}" ({message})', told)

    def look_at(self, ctx, listener):
        """Turn the head to the listener before speaking.

        After memorize_person the head already follows them; otherwise wait
        LOOK_SETTLE for the head to turn, as before judging a person.
        """
        people = People.of(self.runtime)
        track = ctx.memory.get(f"{listener}.track")
        followed = people.following()
        facing = followed is not None and (
            followed[1] == listener or (track is not None and followed[0] == track)
        )
        people.follow(track=track, name=listener)
        message = people.latest()
        if message is None or not any(
            p.visible and (p.id == track or p.name == listener) for p in message.people
        ):
            ctx.log(f"{listener} is not in view; speaking towards the last sighting")
        elif not facing:
            people.pause(people.look_settle, ctx.cancelled)
