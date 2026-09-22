"""Hygiene for anything written into long-term storage.

Learned text is loaded into future prompts, so a poisoned entry keeps steering the agent in
every later conversation. These checks are heuristics, not a security boundary: they catch the
common cases (pasted secrets, "ignore previous instructions", invisible Unicode) cheaply. Put
anything stricter in ``LearnerConfig.extra_review_instructions`` and the approval gate.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width joiner stays: emoji sequences need it and it cannot reorder text. Every other
# format character (zero-width space, bidi overrides/isolates, tag characters) is dropped —
# they let a document read one way to a person and another way to the model (Trojan Source).
_KEEP_FORMAT_CHARS = frozenset("‍")


def sanitize(text: object, *, strip: bool = True) -> str:
    """Normalize newlines and drop invisible format characters. ``strip=False`` keeps edge
    whitespace — a patch fragment's leading/trailing newlines are part of its meaning."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in _KEEP_FORMAT_CHARS or unicodedata.category(ch) != "Cf")
    return text.strip() if strip else text


_SECRET_PATTERNS = [
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"), "an API key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "an AWS access key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "a private key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "a GitHub token"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "a Slack token"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "a Google API key"),
]

_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts?|rules|messages)", re.I),
    re.compile(r"\bdisregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|the\s+system)\s+(?:instructions|prompts?|rules)", re.I),
    re.compile(r"\b(?:reveal|print|output|leak)\s+(?:your|the)\s+(?:system\s+prompt|hidden\s+instructions)", re.I),
    re.compile(r"\bnew\s+system\s+prompt\b", re.I),
    re.compile(r"<\s*/?\s*(?:system|assistant)\s*>", re.I),
]


def scan(text: str) -> str | None:
    """A reason to refuse ``text``, or None when it looks safe to persist."""
    for pattern, label in _SECRET_PATTERNS:
        if pattern.search(text):
            return f"refused: the text contains what looks like {label}; secrets are never stored"
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return "refused: the text contains an instruction aimed at the model (prompt-injection pattern)"
    return None


_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def check_skill_name(name: object) -> str | None:
    if not isinstance(name, str) or not name:
        return "name is required"
    if len(name) > 64:
        return f"name is {len(name)} characters; the limit is 64"
    if not _NAME_RE.match(name):
        return "name must be lowercase letters, digits and single hyphens (e.g. 'deploy-python-service')"
    return None


def check_description(description: str, max_chars: int) -> str | None:
    if not description:
        return "description is required: one sentence saying what the skill is for"
    if "\n" in description:
        return "description must be a single line"
    if len(description) > max_chars:
        return (
            f"description is {len(description)} characters; the limit is {max_chars}. Only the first "
            f"{max_chars} are shown when choosing a skill, so the rest would never help it get used. Shorten it."
        )
    return None
