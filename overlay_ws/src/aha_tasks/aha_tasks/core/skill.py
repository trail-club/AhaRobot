"""Contract between the engine and capability implementations (skills)."""

from dataclasses import dataclass, field
import importlib
import threading


@dataclass
class SkillResult:
    success: bool
    message: str = ""
    # Merged into task memory on success, e.g. {"guest1.name": "Charlie"}.
    outputs: dict = field(default_factory=dict)


@dataclass
class SkillContext:
    subgoal: str
    params: dict
    memory: dict  # copy of the task memory before this subgoal
    cancelled: threading.Event  # set on abort; long-running skills must poll it
    log: object  # log(message) shown in the task status


class Skill:
    """Base class; one instance serves every subgoal bound to the skill name.

    Subclasses are constructed with the runtime (``node``, ``spec``, ``navigation``)
    and must stop promptly once ``ctx.cancelled`` is set.
    """

    def __init__(self, runtime):
        self.runtime = runtime

    @classmethod
    def validate(cls, params, spec):
        """Raise ValueError for unusable params; called before any motion."""

    def ready(self):
        """Block until dependencies are available; raise on timeout."""

    def expected(self, params):
        """State that holds after success; also assumed when the subgoal is mocked."""
        return {}

    def execute(self, ctx):
        """Return a SkillResult (or bool). Exceptions count as failure."""
        raise NotImplementedError


def normalize(result):
    if isinstance(result, SkillResult):
        return result
    if isinstance(result, tuple):
        return SkillResult(bool(result[0]), str(result[1]) if len(result) > 1 else "")
    return SkillResult(bool(result))


def resolve(target):
    """Import ``module:attribute``; ``None`` means the skill is not implemented."""
    if target is None:
        return None
    module, attribute = target.split(":")
    return getattr(importlib.import_module(module), attribute)
