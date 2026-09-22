"""Sessions: frozen context, tool routing, triggers, and awaited background reviews."""

import json
import threading
import time

from musclememory import LearnerConfig, SelfLearner

from .conftest import ScriptedLLM, ops


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def tool_round(i):
    return [{"role": "assistant", "content": None,
             "tool_calls": [{"id": str(i), "function": {"name": "shell", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(i), "content": "ok"}]


def call(session, tool, **args):
    return json.loads(session.handle_tool_call(tool, args))


def test_system_prompt_is_frozen_for_the_session(make_learner):
    learner = make_learner()
    session = learner.session()
    before = session.system_prompt()
    assert call(session, "memory", action="add", target="user", content="Prefers terse answers.")["success"]
    assert session.system_prompt() == before  # cache-stable
    assert "Prefers terse answers." in learner.session().system_prompt()  # next session sees it


def test_tool_formats(make_learner):
    s = make_learner().session()
    openai = s.tools("openai")
    anthropic = s.tools("anthropic")
    assert {t["function"]["name"] for t in openai} == set(s.tool_names)
    assert {t["name"] for t in anthropic} == set(s.tool_names)
    assert all(t["input_schema"]["type"] == "object" for t in anthropic)


def test_skill_patch_requires_view_first(make_learner):
    session = make_learner().session()
    assert call(session, "skill_manage", action="create", name="deploy-service",
                description="Deploy the service.", body="1. make ship")["success"]
    other = session._learner.session()
    refused = call(other, "skill_manage", action="patch", name="deploy-service", old_text="make ship", new_text="x")
    assert not refused["success"] and "skill_view" in refused["message"]
    viewed = call(other, "skill_view", name="deploy-service")
    assert "make ship" in viewed["content"] and "hash" not in viewed
    assert call(other, "skill_manage", action="patch", name="deploy-service", old_text="make ship", new_text="make deploy")["success"]
    # consecutive edits build on the agent's own write
    assert call(other, "skill_manage", action="patch", name="deploy-service", old_text="make deploy", new_text="make go")["success"]


def test_unknown_tool_and_bad_arguments(make_learner):
    session = make_learner().session()
    assert not call(session, "nope")["success"]
    assert "invalid arguments" in json.loads(session.handle_tool_call("memory", "{not json"))["message"]
    assert "no skill named" in call(session, "skill_view", name="missing")["message"]


def test_memory_review_fires_every_n_turns(make_learner):
    learner = make_learner(ops(memory=[{"op": "add", "target": "user", "content": "Likes tea."}]),
                           memory_review_every_turns=3, skill_review_every_tool_rounds=0)
    session = learner.session()
    msgs = []
    futures = []
    for i in range(3):
        msgs += [user(f"turn {i}"), assistant("ok")]
        futures.append(session.end_turn(msgs))
    assert futures[:2] == [None, None]
    result = futures[2].result(timeout=10)
    assert result.scope == "memory" and result.learned
    assert learner.library.memory("user") == ["Likes tea."]


def test_skill_review_counts_tool_rounds_not_turns(make_learner):
    learner = make_learner(memory_review_every_turns=0, skill_review_every_tool_rounds=3)
    session = learner.session()
    msgs = [user("go")] + tool_round(1) + tool_round(2)
    assert session.end_turn(msgs) is None
    msgs += [assistant("done"), user("again")] + tool_round(3)
    future = session.end_turn(msgs)
    assert future is not None and future.result(timeout=10).scope == "skills"


def test_foreground_save_resets_counter(make_learner):
    learner = make_learner(memory_review_every_turns=2, skill_review_every_tool_rounds=0)
    session = learner.session()
    session.end_turn([user("a"), assistant("b")])
    call(session, "memory", action="add", target="user", content="x")
    assert session.end_turn([user("a"), assistant("b"), user("c"), assistant("d")]) is None


def test_close_reviews_unreviewed_activity(make_learner):
    learner = make_learner(ops(memory=[{"op": "add", "target": "user", "content": "Short session fact."}]))
    with learner.session() as session:
        assert session.end_turn([user("I'm a data engineer"), assistant("noted")]) is None
    learner.close()
    assert learner.library.memory("user") == ["Short session fact."]
    assert learner.test_llm.calls, "the final review never ran"


def test_close_without_activity_costs_nothing(make_learner):
    learner = make_learner()
    learner.session().close()
    learner.close()
    assert learner.test_llm.calls == []


def test_learner_close_waits_for_slow_review_and_closes_sessions(make_learner):
    started = threading.Event()

    def slow(system, user):
        started.set()
        time.sleep(0.5)
        return ops(memory=[{"op": "add", "target": "memory", "content": "Slow but landed."}])

    learner = make_learner(slow)
    session = learner.session()
    session.end_turn([user("hi"), assistant("hello")])  # below the interval: only close() reviews it
    assert learner.close(timeout=10)  # closes the open session, then waits
    assert started.is_set() and learner.library.memory("memory") == ["Slow but landed."]


def test_history_rehydrates_counters(make_learner):
    learner = make_learner(memory_review_every_turns=3, skill_review_every_tool_rounds=0)
    history = [user("1"), assistant("a"), user("2"), assistant("b")]
    session = learner.session(history=history)
    assert session.end_turn(history + [user("3"), assistant("c")]) is not None


def test_review_now_with_focus(make_learner):
    learner = make_learner()
    session = learner.session()
    session.end_turn([user("x"), assistant("y")])
    session.review_now(focus="the formatting complaint").result(timeout=10)
    assert "the formatting complaint" in learner.test_llm.calls[0][1]


def test_events_and_review_failures_never_raise(tmp_path):
    events = []

    def broken(system, user):
        raise RuntimeError("provider down")

    learner = SelfLearner(tmp_path / "s", llm=broken, on_event=events.append,
                          config=LearnerConfig(memory_review_every_turns=1))
    result = learner.session().end_turn([user("x"), assistant("y")]).result(timeout=10)
    learner.close()
    assert result.error and "provider down" in result.error
    assert events and events[0]["error"]


def test_no_llm_means_tools_only(tmp_path):
    learner = SelfLearner(tmp_path / "s", config=LearnerConfig(memory_review_every_turns=1))
    session = learner.session()
    assert session.end_turn([user("x"), assistant("y")]) is None
    assert call(session, "memory", action="add", target="user", content="still works")["success"]
    learner.close()


def test_synchronous_mode(make_learner):
    learner = make_learner(ops(memory=[{"op": "add", "target": "user", "content": "sync"}]),
                           background=False, memory_review_every_turns=1)
    future = learner.session().end_turn([user("x"), assistant("y")])
    assert future.done() and learner.library.memory("user") == ["sync"]


def test_recall_for_agents_without_tools(make_learner):
    learner = make_learner()
    s = learner.session()
    call(s, "skill_manage", action="create", name="deploy-service", description="Deploy the Python service.",
         body="1. make ship")
    call(s, "skill_manage", action="create", name="bake-bread", description="Bake sourdough bread.", body="1. knead")
    text = learner.recall("how do I deploy the python service?", k=1)
    assert "make ship" in text and "knead" not in text
    assert learner.recall("quantum chromodynamics") == ""
