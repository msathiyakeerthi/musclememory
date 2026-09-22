"""Tools the agent can call mid-conversation, as neutral JSON Schema plus provider formatters.

Optional: an agent without tool use still learns (the background reviewer writes, and
``SelfLearner.recall`` injects matching skills). Tools let it load skills on demand and save
what it learns while the user is watching.
"""

from __future__ import annotations

TOOL_NAMES = ("memory", "skills_list", "skill_view", "skill_manage")


def tool_specs(description_max_chars: int = 60) -> list[dict]:
    return [
        {
            "name": "memory",
            "description": (
                "Save, update or delete a durable fact in long-term memory. Memory is loaded into every "
                "future conversation, so keep entries to one or two sentences. target 'user': who the user "
                "is and how they want you to work. target 'memory': your own notes about the environment, "
                "conventions and ongoing work. Changes take effect from the next conversation. Never store secrets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                    "target": {"type": "string", "enum": ["user", "memory"]},
                    "content": {"type": "string", "description": "add: the entry. replace: the whole new entry."},
                    "old_text": {"type": "string",
                                 "description": "replace/remove: an exact substring that identifies one existing entry."},
                },
                "required": ["action", "target"],
                "additionalProperties": False,
            },
        },
        {
            "name": "skills_list",
            "description": "List your skills (name and one-line description), including any created in this conversation.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "skill_view",
            "description": (
                "Load a skill's full instructions before doing a task it covers. Pass file_path to open one "
                "of its support files (e.g. references/pitfalls.md)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "file_path": {"type": "string", "description": "Optional support file, e.g. references/pitfalls.md"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "skill_manage",
            "description": (
                "Create or improve a skill: a reusable procedure for a class of task. "
                f"create: name (lowercase-hyphenated, names the class of task), description (one sentence, at most "
                f"{description_max_chars} characters), body (Markdown: steps with exact commands, then pitfalls). "
                "patch: replace old_text (must occur exactly once in SKILL.md) with new_text; call skill_view first. "
                "write_file: add or replace a support file under references/, templates/ or scripts/ "
                "(call skill_view with file_path first to replace an existing one)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "patch", "write_file"]},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "body": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["action", "name"],
                "additionalProperties": False,
            },
        },
    ]


def to_openai(specs: list[dict]) -> list[dict]:
    """Chat Completions ``tools=`` entries."""
    return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                              "parameters": s["parameters"]}} for s in specs]


def to_anthropic(specs: list[dict]) -> list[dict]:
    """Anthropic Messages ``tools=`` entries."""
    return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in specs]


FORMATTERS = {"neutral": lambda specs: specs, "openai": to_openai, "anthropic": to_anthropic}
