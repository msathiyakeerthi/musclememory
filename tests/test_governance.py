"""Approval gate, ledger + rollback, and decay."""

from datetime import timedelta

from musclememory._util import utcnow
from musclememory.cli import main as cli

from .test_library import create, mem, skill


# --- approval ---------------------------------------------------------------------------
def test_gate_stages_agent_writes_and_approve_applies(lib):
    lib.config.write_approval = True
    result = lib.apply(mem("add", content="Prefers terse answers."), "review")
    assert result.staged and lib.memory("user") == []
    assert [p["id"] for p in lib.pending()] == [result.pending_id]
    assert lib.approve(result.pending_id).changed
    assert lib.memory("user") == ["Prefers terse answers."] and lib.pending() == []


def test_gate_does_not_stage_invalid_or_no_op_writes(lib):
    lib.config.write_approval = True
    assert not lib.apply(mem("add", content=""), "review").ok
    assert lib.pending() == []


def test_user_actor_bypasses_gate(lib):
    lib.config.write_approval = True
    assert lib.apply(mem("add", content="x"), "user").changed


def test_approve_revalidates_against_current_state(lib):
    create(lib)
    lib.config.write_approval = True
    h = lib.get_skill("deploy-service").hash
    staged = lib.apply(skill("patch", base_hash=h, old_text="make ship", new_text="make deploy"), "review")
    lib.config.write_approval = False
    lib.apply(skill("patch", base_hash=h, old_text="make ship", new_text="make release"), "review")
    result = lib.approve(staged.pending_id)
    assert not result.ok and "changed since" in result.message


def test_reject(lib):
    lib.config.write_approval = True
    staged = lib.apply(mem("add", content="x"), "review")
    assert lib.reject(staged.pending_id) and lib.pending() == []
    assert not lib.reject("../../etc")


# --- ledger + rollback ------------------------------------------------------------------
def test_rollback_restores_prior_content(lib):
    lib.apply(mem("add", content="first"), "review")
    lib.apply(mem("add", content="second"), "review")
    last = lib.ledger()[-1]
    assert lib.rollback(last["id"]).ok
    assert lib.memory("user") == ["first"]


def test_rollback_fails_closed_when_file_changed_since(lib):
    lib.apply(mem("add", content="first"), "review")
    entry = lib.ledger()[-1]
    lib.apply(mem("add", content="second"), "review")
    result = lib.rollback(entry["id"])
    assert not result.ok and "changed since" in result.message
    assert lib.memory("user") == ["first", "second"]


def test_rollback_of_create_archives_instead_of_deleting(lib):
    create(lib)
    assert lib.rollback(lib.ledger()[-1]["id"]).ok
    assert lib.get_skill("deploy-service") is None
    assert "deploy-service" in lib.store.archived_names()


# --- decay ------------------------------------------------------------------------------
def _age(lib, name, days):
    usage = lib._usage()
    stamp = (utcnow() - timedelta(days=days)).isoformat(timespec="seconds")
    usage[name].update(created_at=stamp, last_used_at=stamp)
    lib._save_usage(usage)


def test_curate_stale_then_archive_then_restore(lib):
    create(lib)
    _age(lib, "deploy-service", 15)
    assert lib.curate()["stale"] == ["deploy-service"]
    _age(lib, "deploy-service", 31)
    assert lib.curate()["archived"] == ["deploy-service"]
    assert lib.get_skill("deploy-service") is None
    assert lib.restore("deploy-service") and lib.skill_info("deploy-service").state == "active"


def test_curate_spares_pinned_and_user_owned(lib):
    create(lib, name="pinned-one")
    create(lib, name="users-own", actor="foreground")
    lib.pin("pinned-one")
    for name in ("pinned-one", "users-own"):
        _age(lib, name, 100)
    assert lib.curate() == {"checked": 0, "stale": [], "archived": []}


def test_use_reactivates_stale_skill(lib):
    create(lib)
    _age(lib, "deploy-service", 15)
    lib.curate()
    lib.view_skill("deploy-service")
    assert lib.skill_info("deploy-service").state == "active"


# --- CLI --------------------------------------------------------------------------------
def test_cli_round_trip(lib, capsys):
    root = str(lib.store.root)
    lib.config.write_approval = True
    staged = lib.apply(mem("add", content="Prefers terse answers."), "review")
    assert cli(["--dir", root, "pending"]) == 0 and staged.pending_id in capsys.readouterr().out
    assert cli(["--dir", root, "approve", "all"]) == 0
    assert cli(["--dir", root, "memory"]) == 0 and "Prefers terse answers." in capsys.readouterr().out
    assert cli(["--dir", root, "status"]) == 0 and "1 entries" in capsys.readouterr().out
    assert cli(["--dir", root, "show", "nope"]) == 1
