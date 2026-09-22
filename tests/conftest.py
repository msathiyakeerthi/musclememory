import json

import pytest

from musclememory import LearnerConfig, SelfLearner
from musclememory.library import Library
from musclememory.store import FileStore

NOTHING = json.dumps({"memory_ops": [], "skill_ops": []})


class ScriptedLLM:
    """Stands in for the reviewer model: returns queued replies and records every prompt."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        reply = self.replies.pop(0) if self.replies else NOTHING
        return reply(system, user) if callable(reply) else reply


def ops(memory=(), skills=(), notes=""):
    return json.dumps({"memory_ops": list(memory), "skill_ops": list(skills), "notes": notes})


@pytest.fixture
def lib(tmp_path):
    return Library(FileStore(tmp_path / "store"), LearnerConfig())


@pytest.fixture
def make_learner(tmp_path):
    made = []

    def factory(*replies, **config):
        llm = ScriptedLLM(*replies)
        learner = SelfLearner(tmp_path / "store", llm=llm, config=LearnerConfig(**config))
        learner.test_llm = llm
        made.append(learner)
        return learner

    yield factory
    for learner in made:
        learner.close(timeout=10)
