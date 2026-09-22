"""Policy over the store: what may be written, by whom, and in what shape.

Every write — from the agent mid-conversation, the background reviewer, or a person at the
CLI — goes through :meth:`Library.apply`, so the rules hold no matter who is writing:

* content is sanitized and scanned before it can reach a future prompt;
* skill descriptions are hard-capped (a long one is silently useless, so it is refused, not cut);
* an existing file can only be changed by someone who read its current version (hash check);
* the autonomous reviewer may only edit skills it created itself, and never pinned ones;
* with ``write_approval`` on, agent writes are staged for a person instead of applied;
* every change is recorded with its prior content, so any single edit can be rolled back.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Callable

from ._util import content_hash, iso, parse_iso, read_text, utcnow
from .config import LearnerConfig
from .guards import check_description, check_skill_name, sanitize, scan
from .store import ENTRY_DELIMITER, MEMORY_FILES, SUPPORT_DIRS, FileStore, Skill, parse_skill, render_skill

logger = logging.getLogger("musclememory")

ACTORS = ("foreground", "review", "user")
# Who created a skill decides who may later change it. Only "learned" skills are the
# reviewer's to edit; a skill the user wrote, or asked the agent to write, stays theirs.
ORIGIN_BY_ACTOR = {"review": "learned", "foreground": "agent", "user": "user"}


@dataclass
class OpResult:
    ok: bool
    message: str
    op: dict = field(default_factory=dict)
    changed: bool = False
    staged: bool = False
    pending_id: str | None = None
    new_hash: str | None = None

    def describe(self) -> str:
        op = self.op
        kind, action = op.get("kind", "?"), op.get("op", "?")
        if kind == "memory":
            text = op.get("content") or op.get("new_text") or op.get("old_text") or ""
            return f"memory {action} [{op.get('target', '?')}]: {_clip(text, 70)}"
        target = op.get("name", "?")
        if action == "write_file":
            target += f"/{op.get('path') or op.get('file_path') or '?'}"
        return f"skill {action}: {target}"

    def to_tool_result(self) -> dict:
        out = {"success": self.ok, "message": self.message}
        if self.staged:
            out["staged_for_approval"] = self.pending_id
        return out


@dataclass
class SkillInfo:
    name: str
    description: str
    origin: str
    state: str
    pinned: bool
    use_count: int = 0
    last_used_at: str | None = None
    created_at: str | None = None

    @property
    def editable_by_reviewer(self) -> bool:
        return self.origin == "learned" and not self.pinned and self.state != "archived"


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class Library:
    def __init__(self, store: FileStore, config: LearnerConfig | None = None):
        self.store = store
        self.config = config or LearnerConfig()
        self._handlers: dict[tuple[str, str], Callable[[dict, str, bool], OpResult]] = {
            ("memory", "add"): self._memory_add,
            ("memory", "replace"): self._memory_replace,
            ("memory", "remove"): self._memory_remove,
            ("skill", "create"): self._skill_create,
            ("skill", "patch"): self._skill_patch,
            ("skill", "write_file"): self._skill_write_file,
        }

    # ================================================================== reads
    def memory(self, target: str) -> list[str]:
        return self.store.read_memory(target)

    def budget(self, target: str) -> int:
        return self.config.user_char_budget if target == "user" else self.config.memory_char_budget

    def _usage(self) -> dict:
        return self.store.load_json("state/usage.json", {})

    def _save_usage(self, usage: dict) -> None:
        self.store.save_json("state/usage.json", usage)

    def _info(self, name: str, skill: Skill | None, usage: dict, *, archived: bool = False) -> SkillInfo:
        u = usage.get(name, {})
        return SkillInfo(
            name=name,
            description=skill.description if skill else "",
            origin=u.get("origin", "user"),  # a skill nobody registered was dropped in by hand
            state="archived" if archived else u.get("state", "active"),
            pinned=bool(u.get("pinned", False)),
            use_count=int(u.get("use_count", 0)),
            last_used_at=u.get("last_used_at"),
            created_at=u.get("created_at"),
        )

    def skill_info(self, name: str) -> SkillInfo | None:
        skill = self.store.read_skill(name)
        return self._info(name, skill, self._usage()) if skill else None

    def skills(self, *, include_archived: bool = False) -> list[SkillInfo]:
        usage = self._usage()
        infos = [self._info(n, self.store.read_skill(n), usage) for n in self.store.skill_names()]
        if include_archived:
            for name in self.store.archived_names():
                meta, _ = parse_skill(read_text(self.store.archive_dir / name / "SKILL.md") or "")
                info = self._info(name, None, usage, archived=True)
                info.description = str(meta.get("description", ""))
                infos.append(info)
        return infos

    def get_skill(self, name: str) -> Skill | None:
        if check_skill_name(name):
            return None
        return self.store.read_skill(name)

    def view_skill(self, name: str, file_path: str | None = None, *, record_use: bool = True) -> dict:
        """Full text of a skill (or one support file) plus the hash an edit must be based on."""
        skill = self.get_skill(name)
        if skill is None:
            raise LookupError(f"no skill named {name!r}; call skills_list to see what exists")
        if file_path:
            rel, err = _support_path(file_path)
            if err:
                raise LookupError(err)
            text = read_text(self.store.skill_path(name, rel))
            if text is None:
                raise LookupError(f"skill {name!r} has no file {rel!r}")
            result = {"name": name, "path": rel, "content": text, "hash": content_hash(text)}
        else:
            result = {"name": name, "path": "SKILL.md", "content": skill.content, "hash": skill.hash,
                      "files": self.store.support_files(name)}
        if record_use:
            self.record_use(name)
        return result

    def record_use(self, name: str) -> None:
        with self.store.lock:
            usage = self._usage()
            u = usage.setdefault(name, {"origin": "user", "state": "active", "pinned": False})
            u["use_count"] = int(u.get("use_count", 0)) + 1
            u["last_used_at"] = iso(utcnow())
            if u.get("state") == "stale":
                u["state"] = "active"
            self._save_usage(usage)

    # ================================================================== writes
    def apply(self, op: dict, actor: str, *, bypass_approval: bool = False) -> OpResult:
        """Validate and perform one operation (see the ``_memory_*`` / ``_skill_*`` handlers)."""
        if actor not in ACTORS:
            raise ValueError(f"actor must be one of {ACTORS}, got {actor!r}")
        op = dict(op or {})
        handler = self._handlers.get((str(op.get("kind")), str(op.get("op"))))
        if handler is None:
            return OpResult(False, f"unknown operation kind={op.get('kind')!r} op={op.get('op')!r}", op)
        with self.store.lock:
            gated = self.config.write_approval and actor != "user" and not bypass_approval
            try:
                result = handler(op, actor, gated)  # gated => validate only
            except Exception as exc:  # a malformed op must never take down the host agent
                logger.exception("musclememory: operation failed")
                return OpResult(False, f"internal error: {exc}", op)
            if gated and result.ok and result.changed:
                pending_id = uuid.uuid4().hex[:12]
                self.store.save_pending(pending_id, {
                    "id": pending_id, "actor": actor, "op": op,
                    "created_at": iso(utcnow()), "summary": result.message,
                })
                return OpResult(True, f"staged for approval as {pending_id}: {result.message}", op,
                                staged=True, pending_id=pending_id)
            return result

    def _commit(self, path, new_text: str | None, actor: str, op: dict, *, action: str) -> str | None:
        """Write (or, with ``new_text=None``, delete) a file and record the change."""
        before = read_text(path)
        if new_text is None:
            path.unlink(missing_ok=True)
        else:
            self.store.write(path, new_text)
        after_hash = content_hash(new_text) if new_text is not None else None
        self.store.append_ledger({
            "id": uuid.uuid4().hex[:12], "ts": iso(utcnow()), "actor": actor, "action": action,
            "path": self.store.rel(path), "summary": OpResult(True, "", op).describe(),
            "before": before, "after_hash": after_hash,
        })
        return after_hash

    # --- memory ------------------------------------------------------------------------
    def _target(self, op: dict) -> tuple[str | None, str | None]:
        target = op.get("target")
        if target not in MEMORY_FILES:
            return None, f"target must be one of {sorted(MEMORY_FILES)}"
        return target, None

    def _clean_entry(self, text: object) -> tuple[str, str | None]:
        text = sanitize(text)
        if not text:
            return "", "content is empty"
        if ENTRY_DELIMITER.strip() in text.splitlines():
            return "", "content may not contain a line holding only '§' (the entry separator)"
        return text, scan(text)

    def _fits(self, target: str, entries: list[str]) -> str | None:
        used, budget = len(ENTRY_DELIMITER.join(entries)), self.budget(target)
        if used > budget:
            return (f"{target} memory would be {used}/{budget} characters. Merge or remove entries "
                    f"(replace/remove) before adding more.")
        return None

    def _find_entry(self, target: str, entries: list[str], old_text: object) -> tuple[int, str | None]:
        old = sanitize(old_text)
        if not old:
            return -1, "old_text is required: an exact substring of the entry to change"
        hits = [i for i, e in enumerate(entries) if old in e]
        if not hits:
            return -1, f"no {target} entry contains {_clip(old, 60)!r}"
        if len(hits) > 1:
            return -1, f"{len(hits)} {target} entries contain {_clip(old, 60)!r}; use a longer, unique substring"
        return hits[0], None

    def _write_memory(self, target: str, entries: list[str], actor: str, op: dict, action: str) -> None:
        self._commit(self.store.memory_path(target), self.store.render_memory(entries), actor, op, action=action)

    def _memory_add(self, op: dict, actor: str, dry_run: bool) -> OpResult:
        target, err = self._target(op)
        if err:
            return OpResult(False, err, op)
        content, err = self._clean_entry(op.get("content"))
        if err:
            return OpResult(False, err, op)
        entries = self.memory(target)
        if _norm(content) in {_norm(e) for e in entries}:
            return OpResult(True, "already in memory; nothing changed", op)
        new = entries + [content]
        if err := self._fits(target, new):
            return OpResult(False, err, op)
        if not dry_run:
            self._write_memory(target, new, actor, op, "memory.add")
        return OpResult(True, f"saved to {target} memory (takes effect next session)", op, changed=True)

    def _memory_replace(self, op: dict, actor: str, dry_run: bool) -> OpResult:
        target, err = self._target(op)
        if err:
            return OpResult(False, err, op)
        entries = self.memory(target)
        idx, err = self._find_entry(target, entries, op.get("old_text"))
        if err:
            return OpResult(False, err, op)
        content, err = self._clean_entry(op.get("new_text", op.get("content")))
        if err:
            return OpResult(False, err + " (use remove to delete an entry)" if content == "" else err, op)
        if any(_norm(content) == _norm(e) for i, e in enumerate(entries) if i != idx):
            return OpResult(False, "another entry already says this; remove this one instead", op)
        new = entries[:idx] + [content] + entries[idx + 1:]
        if err := self._fits(target, new):
            return OpResult(False, err, op)
        if not dry_run:
            self._write_memory(target, new, actor, op, "memory.replace")
        return OpResult(True, f"updated {target} memory entry", op, changed=True)

    def _memory_remove(self, op: dict, actor: str, dry_run: bool) -> OpResult:
        target, err = self._target(op)
        if err:
            return OpResult(False, err, op)
        entries = self.memory(target)
        idx, err = self._find_entry(target, entries, op.get("old_text"))
        if err:
            return OpResult(False, err, op)
        if not dry_run:
            self._write_memory(target, entries[:idx] + entries[idx + 1:], actor, op, "memory.remove")
        return OpResult(True, f"removed {target} memory entry", op, changed=True)

    # --- skills ------------------------------------------------------------------------
    def _check_edit(self, name: str, actor: str, usage: dict) -> str | None:
        u = usage.get(name, {})
        if actor != "review":
            return None
        if u.get("pinned"):
            return f"skill {name!r} is pinned; only the user can change it"
        if u.get("origin", "user") != "learned":
            return (f"skill {name!r} belongs to the user; the reviewer may not edit it. Mention the "
                    f"problem in notes (the user can run `musclememory adopt {name}` to hand it over)")
        return None

    def _check_base(self, actor: str, base_hash: object, current: str, what: str) -> str | None:
        if actor == "user" and not base_hash:
            return None
        if not base_hash:
            return f"base_hash is required to change {what}: edits must be based on the version you read"
        if base_hash != current:
            return (f"{what} changed since you read it (you read {base_hash}, current is {current}); "
                    f"read it again and base your edit on the current text")
        return None

    def _skill_create(self, op: dict, actor: str, dry_run: bool) -> OpResult:
        name = op.get("name")
        if err := check_skill_name(name):
            return OpResult(False, err, op)
        description = sanitize(op.get("description"))
        if err := check_description(description, self.config.description_max_chars):
            return OpResult(False, err, op)
        body = sanitize(op.get("body"))
        if not body:
            return OpResult(False, "body is empty", op)
        if len(body) > self.config.skill_body_max_chars:
            return OpResult(False, f"body is {len(body)} characters; the limit is {self.config.skill_body_max_chars}. "
                                   f"Move depth into references/ files", op)
        if err := scan(description + "\n" + body):
            return OpResult(False, err, op)
        if self.store.read_skill(name):
            return OpResult(False, f"skill {name!r} already exists; patch it instead of creating a duplicate", op)
        if name in self.store.archived_names():
            return OpResult(False, f"skill {name!r} is archived; the user can bring it back with "
                                   f"`musclememory restore {name}`", op)
        content = render_skill(name, description, body)
        if dry_run:
            return OpResult(True, f"would create skill {name!r}", op, changed=True)
        new_hash = self._commit(self.store.skill_path(name), content, actor, op, action="skill.create")
        usage = self._usage()
        now = iso(utcnow())
        usage[name] = {"origin": ORIGIN_BY_ACTOR[actor], "state": "active", "pinned": False,
                       "created_at": now, "last_used_at": now, "use_count": 0}
        self._save_usage(usage)
        return OpResult(True, f"created skill {name!r}", op, changed=True, new_hash=new_hash)

    def _skill_patch(self, op: dict, actor: str, dry_run: bool) -> OpResult:
        name = op.get("name")
        skill = self.get_skill(name) if isinstance(name, str) else None
        if skill is None:
            return OpResult(False, f"no skill named {name!r}", op)
        if err := self._check_edit(name, actor, self._usage()):
            return OpResult(False, err, op)
        if err := self._check_base(actor, op.get("base_hash"), skill.hash, f"skill {name!r}"):
            return OpResult(False, err, op)
        old = op.get("old_text")
        if not isinstance(old, str) or not old:
            return OpResult(False, "old_text is required: exact text that occurs once in SKILL.md", op)
        count = skill.content.count(old)
        if count != 1:
            where = "does not occur" if count == 0 else f"occurs {count} times"
            return OpResult(False, f"old_text {where} in SKILL.md; it must match exactly once", op)
        new_text = op.get("new_text")
        if not isinstance(new_text, str):
            return OpResult(False, "new_text is required (use an empty string to delete old_text)", op)
        new_text = sanitize(new_text, strip=False)
        if err := scan(new_text):
            return OpResult(False, err, op)
        content = skill.content.replace(old, new_text, 1)
        meta, body = parse_skill(content)
        if meta.get("name") != name:
            return OpResult(False, "a patch may not rename the skill or break its frontmatter", op)
        if err := check_description(sanitize(meta.get("description")), self.config.description_max_chars):
            return OpResult(False, err, op)
        if not body:
            return OpResult(False, "the patch would leave the skill body empty", op)
        if len(body) > self.config.skill_body_max_chars:
            return OpResult(False, f"body would be {len(body)} characters; the limit is "
                                   f"{self.config.skill_body_max_chars}. Move depth into references/ files", op)
        if dry_run:
            return OpResult(True, f"would patch skill {name!r}", op, changed=True)
        new_hash = self._commit(self.store.skill_path(name), content, actor, op, action="skill.patch")
        usage = self._usage()
        usage.setdefault(name, {"origin": "user", "state": "active", "pinned": False})["last_patched_at"] = iso(utcnow())
        self._save_usage(usage)
        return OpResult(True, f"patched skill {name!r}", op, changed=True, new_hash=new_hash)

    def _skill_write_file(self, op: dict, actor: str, dry_run: bool) -> OpResult:
        name = op.get("name")
        if not isinstance(name, str) or self.get_skill(name) is None:
            return OpResult(False, f"no skill named {name!r}; create it first", op)
        rel, err = _support_path(op.get("path") or op.get("file_path"))
        if err:
            return OpResult(False, err, op)
        if err := self._check_edit(name, actor, self._usage()):
            return OpResult(False, err, op)
        content = sanitize(op.get("content"))
        if not content:
            return OpResult(False, "content is empty", op)
        if len(content) > self.config.support_file_max_chars:
            return OpResult(False, f"content is {len(content)} characters; the limit is "
                                   f"{self.config.support_file_max_chars}", op)
        if err := scan(content):
            return OpResult(False, err, op)
        path = self.store.skill_path(name, rel)
        existing = read_text(path)
        if existing is not None:
            if err := self._check_base(actor, op.get("base_hash"), content_hash(existing), f"{name}/{rel}"):
                return OpResult(False, err, op)
        if dry_run:
            return OpResult(True, f"would write {name}/{rel}", op, changed=True)
        new_hash = self._commit(path, content + "\n", actor, op, action="skill.write_file")
        return OpResult(True, f"wrote {name}/{rel}", op, changed=True, new_hash=new_hash)

    # ================================================================== governance
    def _set_usage_field(self, name: str, **fields) -> bool:
        with self.store.lock:
            if self.store.read_skill(name) is None:
                return False
            usage = self._usage()
            usage.setdefault(name, {"origin": "user", "state": "active", "pinned": False}).update(fields)
            self._save_usage(usage)
            return True

    def pin(self, name: str, pinned: bool = True) -> bool:
        """Pinned skills are exempt from decay and from every autonomous edit."""
        return self._set_usage_field(name, pinned=pinned)

    def adopt(self, name: str) -> bool:
        """Hand a user-owned skill to the reviewer, so it may keep it up to date."""
        return self._set_usage_field(name, origin="learned")

    def restore(self, name: str) -> bool:
        with self.store.lock:
            if not self.store.restore_skill(name):
                return False
            usage = self._usage()
            usage.setdefault(name, {"origin": "user", "pinned": False}).update(
                state="active", last_used_at=iso(utcnow()))
            self._save_usage(usage)
            self.store.append_ledger({"id": uuid.uuid4().hex[:12], "ts": iso(utcnow()), "actor": "user",
                                      "action": "skill.restore", "path": f"skills/{name}", "summary": f"restore {name}"})
            return True

    def pending(self) -> list[dict]:
        return self.store.list_pending()

    def approve(self, pending_id: str) -> OpResult:
        with self.store.lock:
            item = self.store.load_pending(pending_id)
            if item is None:
                return OpResult(False, f"no pending write {pending_id!r}")
            result = self.apply(item["op"], item.get("actor", "review"), bypass_approval=True)
            self.store.delete_pending(pending_id)
            return result

    def reject(self, pending_id: str) -> bool:
        with self.store.lock:
            return self.store.delete_pending(pending_id)

    def ledger(self, limit: int | None = None) -> list[dict]:
        entries = self.store.read_ledger()
        return entries[-limit:] if limit else entries

    def rollback(self, entry_id: str) -> OpResult:
        """Undo one recorded change — only if nothing has touched the file since (fails closed)."""
        with self.store.lock:
            entry = next((e for e in self.store.read_ledger() if e.get("id") == entry_id), None)
            if entry is None or "after_hash" not in entry:
                return OpResult(False, f"no rollback-able ledger entry {entry_id!r}")
            path = self.store.root / entry["path"]
            current = read_text(path)
            current_hash = content_hash(current) if current is not None else None
            if current_hash != entry["after_hash"]:
                return OpResult(False, f"{entry['path']} has changed since {entry_id}; roll back later "
                                       f"changes first (see `musclememory ledger`)")
            op = {"kind": "rollback", "op": entry_id, "name": entry["path"]}
            if entry.get("before") is not None:
                self._commit(path, entry["before"], "user", op, action=f"rollback:{entry_id}")
            elif path.name == "SKILL.md":
                # Undoing a creation archives the skill rather than deleting it.
                self.store.archive_skill(path.parent.name)
                self.store.append_ledger({"id": uuid.uuid4().hex[:12], "ts": iso(utcnow()), "actor": "user",
                                          "action": f"rollback:{entry_id}", "path": entry["path"],
                                          "summary": f"archived {path.parent.name}"})
            else:
                self._commit(path, None, "user", op, action=f"rollback:{entry_id}")
            return OpResult(True, f"rolled back {entry_id} ({entry.get('summary', entry['path'])})", op, changed=True)

    # ================================================================== decay
    def curate(self, now: datetime | None = None) -> dict:
        """Age learned skills by disuse: active → stale → archived. Never deletes; never touches
        pinned skills or skills the user owns."""
        now = now or utcnow()
        report = {"checked": 0, "stale": [], "archived": []}
        with self.store.lock:
            usage = self._usage()
            for name in self.store.skill_names():
                u = usage.get(name)
                if not u or u.get("origin") != "learned" or u.get("pinned"):
                    continue
                report["checked"] += 1
                last = max(filter(None, (parse_iso(u.get(k)) for k in
                                         ("last_used_at", "last_patched_at", "created_at"))), default=now)
                idle_days = (now - last).total_seconds() / 86400
                if idle_days >= self.config.archive_after_days:
                    self.store.archive_skill(name)
                    u["state"] = "archived"
                    report["archived"].append(name)
                    self.store.append_ledger({"id": uuid.uuid4().hex[:12], "ts": iso(now), "actor": "curator",
                                              "action": "skill.archive", "path": f"skills/{name}",
                                              "summary": f"archived {name} after {idle_days:.0f} idle days"})
                elif idle_days >= self.config.stale_after_days and u.get("state", "active") == "active":
                    u["state"] = "stale"
                    report["stale"].append(name)
            self._save_usage(usage)
            self.store.save_json("state/curator.json", {"last_run_at": iso(now), "last_report": report})
        return report

    def curate_due(self, now: datetime | None = None) -> bool:
        last = parse_iso(self.store.load_json("state/curator.json", {}).get("last_run_at"))
        return last is None or ((now or utcnow()) - last).total_seconds() >= self.config.curate_every_days * 86400


def _support_path(path: object) -> tuple[str, str | None]:
    if not isinstance(path, str) or not path.strip():
        return "", "path is required (e.g. references/pitfalls.md)"
    p = PurePosixPath(path.strip().replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or len(p.parts) < 2 or p.parts[0] not in SUPPORT_DIRS:
        return "", f"path must be inside {', '.join(d + '/' for d in SUPPORT_DIRS)} (e.g. references/pitfalls.md)"
    return p.as_posix(), None


def ops_to_dicts(results: list[OpResult]) -> list[dict]:
    return [asdict(r) for r in results]
