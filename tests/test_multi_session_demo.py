"""The multi-session demo is also a regression test: it exercises a full profile lifecycle
(twelve conversations, six sessions, create + patch + nothing-to-save) against the real store."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "multi_session_demo.py"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("multi_session_demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def profile(demo, tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("demo") / "profile"
    assert demo.main(["--dir", str(root), "--quiet"]) == 0, "the demo's own checks failed"
    return root


def test_preferences_became_memory(demo, profile):
    from musclememory.library import Library
    from musclememory.store import FileStore

    entries = Library(FileStore(profile)).memory("user")
    assert any("3 short bullets" in e for e in entries)
    assert any("Friday" in e for e in entries)


def test_procedures_became_skills(demo, profile):
    from musclememory.library import Library
    from musclememory.store import FileStore

    skills = {s.name: s for s in Library(FileStore(profile)).skills()}
    assert set(skills) == {"deploy-python-service", "fix-flaky-integration-tests", "write-release-notes"}
    assert all(s.origin == "learned" for s in skills.values())


def test_a_stale_step_was_patched_not_appended(demo, profile):
    """After the release script started requiring VERSION, the skill's own step must change."""
    body = (profile / "skills" / "deploy-python-service" / "SKILL.md").read_text(encoding="utf-8")
    assert "VERSION=" in body
    assert "UPDATE:" not in body


def test_the_demo_is_idempotent_about_a_used_profile(demo, profile):
    assert demo.main(["--dir", str(profile), "--quiet"]) == 1, "it must refuse to reuse a profile"
