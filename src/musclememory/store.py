"""File-backed persistence: plain Markdown and JSON that a person can read, edit and version.

Layout under ``root``::

    memory/USER.md, memory/MEMORY.md   entries separated by a line holding only "§"
    skills/<name>/SKILL.md             frontmatter + Markdown body; optional references/ templates/ scripts/
    skills/.archive/<name>/            archived skills (never deleted)
    state/usage.json                   per-skill origin, lifecycle state, pin and usage counters
    state/ledger.jsonl                 one line per change, with the prior content for rollback
    state/pending/<id>.json            writes staged for approval
    state/curator.json                 when decay last ran

Writes are atomic (temp file + rename). One process should own a root at a time; inside a
process ``lock`` serializes writers, and content hashes catch edits made underneath a reader.
For a multi-user app, give each user (or tenant) their own root.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ._util import atomic_write, content_hash, read_text, utcnow

logger = logging.getLogger("musclememory")

MEMORY_FILES = {"user": "USER.md", "memory": "MEMORY.md"}
ENTRY_DELIMITER = "\n§\n"
SUPPORT_DIRS = ("references", "templates", "scripts")
_PENDING_ID = re.compile(r"^[0-9a-f]{6,32}$")
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    content: str  # the whole SKILL.md, frontmatter included
    hash: str
    meta: dict = field(default_factory=dict)


def parse_skill(text: str) -> tuple[dict, str]:
    """Split SKILL.md into (frontmatter fields, body). Only flat ``key: value`` lines are read."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text.strip()
    meta: dict = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if not sep or not key.strip() or key.startswith((" ", "\t")):
            continue
        value = value.strip()
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except ValueError:
                value = value.strip('"')
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        meta[key.strip()] = value
    return meta, text[match.end():].strip()


def render_skill(name: str, description: str, body: str) -> str:
    # json.dumps yields a valid YAML double-quoted scalar, so colons and quotes are safe.
    return f"---\nname: {name}\ndescription: {json.dumps(description, ensure_ascii=False)}\n---\n\n{body.strip()}\n"


class FileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.lock = threading.RLock()
        for sub in ("memory", "skills", "state/pending"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # --- paths ---------------------------------------------------------------------------
    def memory_path(self, target: str) -> Path:
        return self.root / "memory" / MEMORY_FILES[target]

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    @property
    def archive_dir(self) -> Path:
        return self.skills_dir / ".archive"

    def skill_path(self, name: str, relpath: str = "SKILL.md") -> Path:
        return self.skills_dir / name / relpath

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # --- memory --------------------------------------------------------------------------
    def read_memory(self, target: str) -> list[str]:
        text = read_text(self.memory_path(target)) or ""
        return [entry.strip() for entry in text.split(ENTRY_DELIMITER) if entry.strip()]

    @staticmethod
    def render_memory(entries: list[str]) -> str:
        return ENTRY_DELIMITER.join(entries) + "\n" if entries else ""

    # --- skills --------------------------------------------------------------------------
    def skill_names(self) -> list[str]:
        return sorted(
            p.name for p in self.skills_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / "SKILL.md").is_file()
        )

    def archived_names(self) -> list[str]:
        if not self.archive_dir.is_dir():
            return []
        return sorted(p.name for p in self.archive_dir.iterdir() if (p / "SKILL.md").is_file())

    def read_skill(self, name: str) -> Skill | None:
        text = read_text(self.skill_path(name))
        if text is None:
            return None
        meta, body = parse_skill(text)
        return Skill(name=name, description=str(meta.get("description", "")), body=body,
                     content=text, hash=content_hash(text), meta=meta)

    def support_files(self, name: str) -> list[str]:
        base = self.skills_dir / name
        return sorted(
            p.relative_to(base).as_posix()
            for sub in SUPPORT_DIRS if (base / sub).is_dir()
            for p in (base / sub).rglob("*") if p.is_file()
        )

    def archive_skill(self, name: str) -> Path:
        src = self.skills_dir / name
        dest = self.archive_dir / name
        if dest.exists():
            dest = self.archive_dir / f"{name}.{utcnow().strftime('%Y%m%d%H%M%S')}"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        return dest

    def restore_skill(self, name: str) -> bool:
        src, dest = self.archive_dir / name, self.skills_dir / name
        if not (src / "SKILL.md").is_file() or dest.exists():
            return False
        shutil.move(str(src), str(dest))
        return True

    # --- generic text + JSON -------------------------------------------------------------
    def write(self, path: Path, text: str) -> None:
        atomic_write(path, text)

    def load_json(self, relpath: str, default):
        text = read_text(self.root / relpath)
        if text is None:
            return default
        try:
            return json.loads(text)
        except ValueError:
            logger.warning("musclememory: %s is not valid JSON; ignoring it", relpath)
            return default

    def save_json(self, relpath: str, data) -> None:
        atomic_write(self.root / relpath, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    # --- ledger --------------------------------------------------------------------------
    def append_ledger(self, entry: dict) -> None:
        path = self.root / "state" / "ledger.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_ledger(self) -> list[dict]:
        text = read_text(self.root / "state" / "ledger.jsonl") or ""
        entries = []
        for line in text.splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
        return entries

    # --- pending approvals ---------------------------------------------------------------
    def save_pending(self, pending_id: str, data: dict) -> None:
        self.save_json(f"state/pending/{pending_id}.json", data)

    def load_pending(self, pending_id: str) -> dict | None:
        if not _PENDING_ID.match(pending_id):
            return None
        return self.load_json(f"state/pending/{pending_id}.json", None)

    def delete_pending(self, pending_id: str) -> bool:
        if not _PENDING_ID.match(pending_id):
            return False
        path = self.root / "state" / "pending" / f"{pending_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_pending(self) -> list[dict]:
        items = [self.load_json(self.rel(p), None) for p in (self.root / "state" / "pending").glob("*.json")]
        return sorted((i for i in items if i), key=lambda i: i.get("created_at", ""))
