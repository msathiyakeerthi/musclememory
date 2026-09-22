"""The reviewer: JSON in, guarded writes out."""

import json

from musclememory.review import extract_json, run_review, select_skills
from musclememory.transcript import normalize

from .conftest import ScriptedLLM, ops
from .test_library import create

CONVO = normalize([
    {"role": "user", "content": "Deploy the service. And please keep answers short."},
    {"role": "assistant", "content": "Deployed with make ship."},
])


def test_applies_memory_and_skill_ops(lib):
    llm = ScriptedLLM(ops(
        memory=[{"op": "add", "target": "user", "content": "Wants short answers."}],
        skills=[{"op": "create", "name": "deploy-service", "description": "Deploy the service.",
                 "body": "1. Run `make ship`."}],
        notes="fine",
    ))
    result = run_review(llm, lib, CONVO, scope="both")
    assert result.learned and not result.rejected and result.notes == "fine"
    assert lib.memory("user") == ["Wants short answers."]
    assert lib.skill_info("deploy-service").origin == "learned"
    system, user = llm.calls[0]
    assert "DO NOT CAPTURE" in system and "make ship" in user


def test_scope_filters_ops(lib):
    llm = ScriptedLLM(ops(memory=[{"op": "add", "target": "user", "content": "x"}],
                          skills=[{"op": "create", "name": "a", "description": "A.", "body": "b"}]))
    run_review(llm, lib, CONVO, scope="memory")
    assert lib.memory("user") == ["x"] and lib.get_skill("a") is None
    assert "memory only" in llm.calls[0][0] and "SKILLS hold" not in llm.calls[0][0]


def test_repair_round_fixes_rejected_op(lib):
    long_desc = "A skill that deploys the Python service to production using the make ship target."
    llm = ScriptedLLM(
        ops(skills=[{"op": "create", "name": "deploy-service", "description": long_desc, "body": "1. make ship"}]),
        ops(skills=[{"op": "create", "name": "deploy-service", "description": "Deploy the service.", "body": "1. make ship"}]),
    )
    result = run_review(llm, lib, CONVO)
    assert result.learned and not result.rejected
    assert "the limit is 60" in llm.calls[1][1]  # the error was shown back to the reviewer


def test_unfixed_rejections_are_reported(lib):
    bad = ops(skills=[{"op": "create", "name": "Bad Name", "description": "x.", "body": "b"}])
    result = run_review(ScriptedLLM(bad, bad), lib, CONVO)
    assert not result.learned and len(result.rejected) == 1 and "lowercase" in result.rejected[0].message


def test_non_json_reply_gets_one_retry(lib):
    llm = ScriptedLLM("Sure! I think we should save...", ops(memory=[{"op": "add", "target": "user", "content": "x"}]))
    assert run_review(llm, lib, CONVO).learned
    assert "could not be parsed" in llm.calls[1][1]
    result = run_review(ScriptedLLM("nope", "still nope"), lib, CONVO)
    assert result.error and "JSON" in result.error


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": "brace } in \\" string"} done') == {"a": 'brace } in " string'}


def test_reviewer_sees_skill_hash_and_can_patch(lib):
    create(lib)
    h = lib.get_skill("deploy-service").hash

    def reply(system, user):
        assert f"base_hash: {h}" in user
        return ops(skills=[{"op": "patch", "name": "deploy-service", "base_hash": h,
                            "old_text": "make ship", "new_text": "make ship --prod"}])

    assert run_review(ScriptedLLM(reply), lib, CONVO, viewed=["deploy-service"]).learned
    assert "--prod" in lib.get_skill("deploy-service").body


def test_select_skills_prefers_viewed_then_lexical(lib):
    create(lib, name="deploy-service", description="Deploy the service.")
    create(lib, name="write-reports", description="Write weekly reports.")
    create(lib, name="bake-bread", description="Bake sourdough bread.")
    names = [s.name for s, _ in select_skills(lib, "please deploy the service", ["bake-bread"], 2)]
    assert names == ["bake-bread", "deploy-service"]


def test_extra_instructions_reach_the_reviewer(lib):
    lib.config.extra_review_instructions = "Never store order IDs."
    llm = ScriptedLLM()
    run_review(llm, lib, CONVO)
    assert "Never store order IDs." in llm.calls[0][0]


def test_protected_skill_problem_goes_to_notes(lib):
    create(lib, actor="foreground")
    h = lib.get_skill("deploy-service").hash
    patch = ops(skills=[{"op": "patch", "name": "deploy-service", "base_hash": h, "old_text": "make ship", "new_text": "x"}])
    result = run_review(ScriptedLLM(patch, json.dumps({"memory_ops": [], "skill_ops": [], "notes": "user skill is outdated"})),
                        lib, CONVO, viewed=["deploy-service"])
    assert not result.learned and result.notes == "user skill is outdated" and not result.rejected
