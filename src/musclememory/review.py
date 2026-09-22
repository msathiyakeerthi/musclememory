"""The reviewer: one LLM call over a finished conversation, turned into guarded writes.

Portable by design — the model only has to return JSON, so any chat model works (no tool-use or
structured-output support required). Every operation it proposes goes through the same
:meth:`Library.apply` rules as a person would; rejected ones are shown back to it once so it can
fix them (the classic case: a description over 60 characters).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ._util import overlap_score, tokens
from .library import Library, OpResult
from .prompts import build_repair, build_system, build_user
from .transcript import Turn, render

logger = logging.getLogger("musclememory")

LLM = Callable[[str, str], str]


@dataclass
class ReviewResult:
    scope: str
    applied: list[OpResult] = field(default_factory=list)
    rejected: list[OpResult] = field(default_factory=list)
    notes: str = ""
    error: str | None = None

    @property
    def learned(self) -> bool:
        return any(r.changed or r.staged for r in self.applied)

    def summary(self) -> str:
        if self.error:
            return f"review failed: {self.error}"
        done = [r.describe() + (" (staged)" if r.staged else "") for r in self.applied if r.changed or r.staged]
        text = "learned: " + "; ".join(done) if done else "nothing to save"
        if self.rejected:
            text += f" ({len(self.rejected)} rejected)"
        return text

    def to_event(self) -> dict:
        return {
            "type": "review", "scope": self.scope, "learned": self.learned, "summary": self.summary(),
            "applied": [r.describe() for r in self.applied if r.changed or r.staged],
            "staged": [r.pending_id for r in self.applied if r.staged],
            "rejected": [{"op": r.describe(), "error": r.message} for r in self.rejected],
            "notes": self.notes, "error": self.error,
        }


_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$", re.S)


def extract_json(text: str) -> dict:
    """The first JSON object in a reply, tolerating code fences and prose around it."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty reply")
    text = text.strip()
    match = _FENCE.match(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        data = _first_object(text)
    if not isinstance(data, dict):
        raise ValueError("the reply is JSON but not an object")
    return data


def _first_object(text: str):
    start = text.find("{")
    while start != -1:
        depth, in_str, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in the reply")


def collect_ops(data: dict, scope: str) -> tuple[list[dict], list[OpResult]]:
    ops, malformed = [], []
    sources = []
    if scope in ("memory", "both"):
        sources.append(("memory", data.get("memory_ops")))
    if scope in ("skills", "both"):
        sources.append(("skill", data.get("skill_ops")))
    for kind, items in sources:
        if items is None:
            continue
        if not isinstance(items, list):
            malformed.append(OpResult(False, f"{kind}_ops must be a list", {"kind": kind}))
            continue
        for item in items:
            if isinstance(item, dict):
                ops.append({**item, "kind": kind})
            else:
                malformed.append(OpResult(False, "each operation must be a JSON object", {"kind": kind}))
    return ops, malformed


def select_skills(library: Library, transcript: str, viewed: Iterable[str], limit: int) -> list:
    """Skills to show in full: those loaded during the conversation first (the likeliest to need
    fixing), then the ones whose name/description best match what was discussed."""
    infos = {s.name: s for s in library.skills() if s.state != "archived"}
    chosen = [n for n in dict.fromkeys(viewed) if n in infos][:limit]
    if len(chosen) < limit:
        query = tokens(transcript)
        scored = sorted(
            ((overlap_score(query, tokens(f"{n.replace('-', ' ')} {i.description}")), n)
             for n, i in infos.items() if n not in chosen),
            reverse=True,
        )
        chosen += [n for score, n in scored if score > 0][: limit - len(chosen)]
    pairs = []
    for name in chosen:
        skill = library.get_skill(name)
        if skill is not None:
            pairs.append((skill, infos[name]))
    return pairs


def run_review(llm: LLM, library: Library, turns: list[Turn], *, scope: str = "both",
               viewed: Iterable[str] = (), focus: str | None = None) -> ReviewResult:
    cfg = library.config
    viewed = list(viewed)
    transcript = render(turns, cfg.review_transcript_max_chars)
    system = build_system(scope, desc_max=cfg.description_max_chars, extra=cfg.extra_review_instructions)
    result = ReviewResult(scope=scope)
    repair = ""
    for _round in range(1 + max(0, cfg.review_repair_rounds)):
        in_play = select_skills(library, transcript, viewed, cfg.review_max_skill_bodies) if scope != "memory" else []
        user = build_user(library, transcript, scope, in_play, focus) + repair
        reply = llm(system, user)
        try:
            data = extract_json(reply)
        except ValueError as exc:
            reply = llm(system, user + f"\n\nYour previous reply could not be parsed ({exc}). "
                                       "Reply with the JSON object only: no prose, no code fences.")
            try:
                data = extract_json(reply)
            except ValueError as exc2:
                result.error = f"the reviewer did not return a JSON object ({exc2})"
                return result
        if isinstance(data.get("notes"), str) and data["notes"].strip():
            result.notes = (result.notes + "\n" + data["notes"].strip()).strip()
        ops, rejected = collect_ops(data, scope)
        for op in ops:
            outcome = library.apply(op, "review")
            (result.applied if outcome.ok else rejected).append(outcome)
        result.rejected = rejected
        if not rejected:
            break
        repair = build_repair(reply, rejected)
    return result
