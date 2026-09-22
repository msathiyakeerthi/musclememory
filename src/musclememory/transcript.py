"""Read any agent's message list into one neutral shape.

Accepted, freely mixed, as dicts or SDK objects:

* OpenAI chat: ``{"role": "assistant", "tool_calls": [{"function": {"name", "arguments"}}]}``,
  ``{"role": "tool", "content": ...}``
* Anthropic Messages: content blocks ``text`` / ``tool_use`` / ``tool_result`` (dicts, or the
  SDK's typed blocks from ``response.content``)
* LangChain messages: ``.type`` in human/ai/tool/system, ``AIMessage.tool_calls``
* plain ``{"role": ..., "content": "..."}``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

_ROLE_ALIASES = {
    "human": "user", "ai": "assistant", "model": "assistant", "function": "tool",
    "developer": "system",
}
_SKIP_BLOCKS = {"thinking", "redacted_thinking", "image", "document", "input_image", "image_url"}


@dataclass
class Turn:
    role: str  # user | assistant | tool | system
    text: str = ""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (name, arguments as text)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _args_text(args) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(args)


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif _get(item, "type") not in _SKIP_BLOCKS:
                parts.append(_get(item, "text") or "")
        return "\n".join(p for p in parts if p)
    text = _get(value, "text")
    return text if isinstance(text, str) else _args_text(value)


def normalize(messages) -> list[Turn]:
    turns: list[Turn] = []
    for m in messages or []:
        role = str(_get(m, "role") or _get(m, "type") or "").lower()
        role = _ROLE_ALIASES.get(role, role)
        content = _get(m, "content")
        texts: list[str] = []
        calls: list[tuple[str, str]] = []
        is_tool_result = role == "tool"
        if isinstance(content, (list, tuple)):
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                    continue
                btype = _get(block, "type")
                if btype == "tool_use":
                    calls.append((str(_get(block, "name") or "?"), _args_text(_get(block, "input"))))
                elif btype == "tool_result":
                    is_tool_result = True
                    texts.append(_as_text(_get(block, "content")))
                elif btype not in _SKIP_BLOCKS:
                    texts.append(_get(block, "text") or "")
        else:
            texts.append(_as_text(content))
        for call in _get(m, "tool_calls") or []:
            fn = _get(call, "function")
            if fn is not None:
                calls.append((str(_get(fn, "name") or "?"), _args_text(_get(fn, "arguments"))))
            else:
                calls.append((str(_get(call, "name") or "?"),
                              _args_text(_get(call, "args", _get(call, "arguments")))))
        text = "\n".join(t for t in texts if t).strip()
        turns.append(Turn("tool" if is_tool_result else role, text, calls))
    return turns


def count_tool_rounds(turns: list[Turn]) -> int:
    """Assistant messages that called at least one tool — Hermes' "tool iterations"."""
    return sum(1 for t in turns if t.role == "assistant" and t.tool_calls)


def count_user_turns(turns: list[Turn]) -> int:
    return sum(1 for t in turns if t.role == "user")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f" …[{len(text) - limit} more chars]"


def render(turns: list[Turn], max_chars: int) -> str:
    """Transcript text for the reviewer, keeping the most recent part when it is too long."""
    lines = []
    for t in turns:
        if t.role == "system":
            continue  # the host's system prompt, which already contains the learned context
        if t.role == "tool":
            lines.append(f"[tool result] {_clip(t.text, 800)}")
            continue
        if t.text:
            lines.append(f"[{t.role}] {t.text}")
        for name, args in t.tool_calls:
            lines.append(f"[{t.role} -> tool] {name}({_clip(args, 400)})")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        text = f"[… {len(text) - max_chars} earlier characters omitted …]\n\n" + text[-max_chars:]
    return text
