"""Small shared helpers: hashing, time, atomic file writes, lexical matching."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def content_hash(text: str) -> str:
    """Short, stable fingerprint of a file's text — the unit of read-before-write checks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def read_text(path: Path) -> str | None:
    """File text with universal newlines (CRLF on disk reads back as LF), or None if absent."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def atomic_write(path: Path, text: str) -> None:
    """Write via temp file + rename so a crash never leaves a half-written store.

    Always LF on disk: the memory delimiter is newline-sensitive, and a CRLF file would stop
    parsing into entries on Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                # Windows: an antivirus scanner or a reader can hold the target open briefly.
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


_STOPWORDS = frozenset(
    "the and for with that this from have are was were you your our can will not but all any "
    "use using into then than when what which how who why about after before there their them "
    "they its also just more most some such only over under out get got let one two".split()
)


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3 and t not in _STOPWORDS}


def overlap_score(query: set[str], doc: set[str]) -> float:
    """Cheap relevance: shared terms, damped by document length. Zero means unrelated."""
    if not query or not doc:
        return 0.0
    return len(query & doc) / math.sqrt(len(doc) + 1)
