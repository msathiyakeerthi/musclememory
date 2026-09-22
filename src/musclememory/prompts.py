"""Prompt text: the reviewer's instructions, and the learned-context block for the agent.

The reviewer rules are distilled from Hermes Agent's background review (MIT, Nous Research):
class-level skills, rules-with-mechanism instead of incident narratives, and — most
important — a do-not-capture list, because an agent that writes down "tool X is broken"
will refuse to use X long after it was fixed.
"""

from __future__ import annotations

from .store import ENTRY_DELIMITER

SCOPES = ("memory", "skills", "both")

_INTRO = """\
You are the learning reviewer for an AI agent. You read a conversation the agent has just had \
and decide what, if anything, it should carry into FUTURE conversations. Nobody reads your reply \
except the program that applies it. Reply with exactly one JSON object and nothing else."""

_MEMORY_STORE = """\
MEMORY holds short facts loaded into EVERY future conversation.
  target "user"   - who the user is: role, expertise, preferences, and standing expectations of how the agent should work.
  target "memory" - the agent's own notes: facts about the environment, project conventions, the state of long-running work.
  One or two sentences per entry. There is a size budget: merge and replace rather than pile up."""

_SKILL_STORE = """\
SKILLS hold procedures for a CLASS of task, loaded only when a future task matches.
  name        - lowercase-hyphenated; names the class of task ("deploy-python-service"), never today's instance ("fix-bug-in-pr-42").
  description - ONE sentence of at most {desc_max} characters. Only these characters are shown when the agent picks a skill; a longer description is rejected.
  body        - Markdown: the steps in the order they are done, with exact commands and decision points; pitfalls attached to the step they affect."""

_WHEN_MEMORY = """\
WHEN TO WRITE MEMORY
- The user revealed something durable about themselves: role, expertise, preferences, constraints.
- The user said how they want the agent to work ("keep answers short", "always ask before deleting").
- A fact about the environment will matter again: paths, available tools, conventions.
Not: things that only matter inside this conversation, or anything memory already says (replace an entry to update it)."""

_WHEN_SKILL = """\
WHEN TO WRITE A SKILL (any one of these is a signal)
- The user corrected the agent's approach, format, tone or verbosity. Put the correction into the skill that governs that kind of task, so the next conversation starts already knowing.
- The agent found a non-trivial technique, fix, or sequence of steps after trial and error.
- A skill used in this conversation turned out to be wrong, incomplete or outdated. Fix it."""

_HOW_SKILL = """\
HOW TO WRITE IT
- Prefer, in this order: patch a skill that was used in this conversation; patch an existing skill that covers the class of task; add a references/ file under an existing skill; only then create a new skill.
- A pitfall is a general rule plus one clause of WHY (the mechanism), in the imperative: "Run migrations before seeding - the seed script assumes the new columns exist." Not a story of what happened today.
- No dates, ticket or PR numbers, or quotes of the user as content. The rule must make sense without today's conversation behind it.
- Write only what this conversation showed working or the user stated. Do not pad a skill with advice from general knowledge: a future session will trust every line as tested.
- The same lesson twice is one rule: strengthen the existing sentence instead of adding a copy. When a skill is wrong, edit the sentence that misled; never append "UPDATE:" below it.
- Support files: references/<topic>.md for depth needed only sometimes, templates/<name> for files to copy and adapt, scripts/<name> for scripts to run. Give SKILL.md a one-line pointer to any new support file."""

_DO_NOT_CAPTURE = """\
DO NOT CAPTURE (these harden into false rules the agent keeps obeying after they stop being true)
- Failures caused by the environment: a missing binary or package, missing credentials, a wrong path. If the fix was found, capture the FIX ("install X with ..."), never "X doesn't work".
- Claims that a tool or feature is broken or unavailable.
- Transient errors that went away on retry. The lesson, if any, is the retry pattern.
- One-off tasks that are not a repeatable class of work.
- Approaches that never worked. If the conversation ended without a working method, save nothing about it.
- Secrets, credentials or API keys, ever. Sensitive personal data unless the user explicitly asked to have it remembered.

Saving nothing is the right answer when the conversation had no corrections, revealed nothing durable and produced no new technique. Otherwise act."""

_OUTPUT_HEAD = """\
OUTPUT - one JSON object:
{
  "memory_ops": [...],
  "skill_ops": [...],
  "notes": "optional: overlapping skills, or user-owned skills that look wrong"
}"""

_OUTPUT_MEMORY = """\
memory_ops items:
  {"op": "add",     "target": "user"|"memory", "content": "the entry"}
  {"op": "replace", "target": "user"|"memory", "old_text": "exact, unique substring of ONE entry", "new_text": "the whole new entry"}
  {"op": "remove",  "target": "user"|"memory", "old_text": "exact, unique substring of ONE entry"}"""

_OUTPUT_SKILLS = """\
skill_ops items:
  {"op": "create",     "name": "...", "description": "...", "body": "..."}
  {"op": "patch",      "name": "...", "base_hash": "the hash shown with that skill", "old_text": "exact text occurring once in SKILL.md", "new_text": "replacement"}
  {"op": "write_file", "name": "...", "path": "references/<topic>.md", "content": "..."}
Patch only skills shown in full below and marked editable, using the base_hash shown. Skills marked read-only belong to the user: report problems with them in "notes" instead."""


def build_system(scope: str, *, desc_max: int, extra: str = "") -> str:
    memory, skills = scope in ("memory", "both"), scope in ("skills", "both")
    parts = [_INTRO]
    if memory:
        parts.append(_MEMORY_STORE)
    if skills:
        parts.append(_SKILL_STORE.format(desc_max=desc_max))
    if memory:
        parts.append(_WHEN_MEMORY)
    if skills:
        parts += [_WHEN_SKILL, _HOW_SKILL]
    parts.append(_DO_NOT_CAPTURE)
    parts.append(_OUTPUT_HEAD)
    if memory:
        parts.append(_OUTPUT_MEMORY)
    if skills:
        parts.append(_OUTPUT_SKILLS)
    if not memory:
        parts.append('This review covers skills only: "memory_ops" must be [].')
    if not skills:
        parts.append('This review covers memory only: "skill_ops" must be [].')
    if extra.strip():
        parts.append("ADDITIONAL RULES FROM THE DEVELOPER (these override the above)\n" + extra.strip())
    return "\n\n".join(parts)


def build_user(library, transcript: str, scope: str, in_play: list, focus: str | None = None) -> str:
    """The reviewer's working context: current stores, the skills in play, and the conversation.

    ``in_play`` is a list of ``(Skill, SkillInfo)``; their full text and hash are shown so every
    patch is based on the version that is actually on disk.
    """
    sections = []
    if scope in ("memory", "both"):
        lines = ["## Current memory"]
        for target in ("user", "memory"):
            entries = library.memory(target)
            used = len(ENTRY_DELIMITER.join(entries))
            lines.append(f'### target "{target}" ({used}/{library.budget(target)} characters used)')
            lines += [f"[{i}] {e}" for i, e in enumerate(entries, 1)] or ["(empty)"]
        sections.append("\n".join(lines))
    if scope in ("skills", "both"):
        index = [s for s in library.skills() if s.state != "archived"]
        lines = ["## Skill index"]
        lines += [f"- {s.name} [{'editable' if s.editable_by_reviewer else 'read-only'}]: {s.description}"
                  for s in index] or ["(no skills yet)"]
        sections.append("\n".join(lines))
        if in_play:
            blocks = ["## Skills relevant to this conversation (full text)"]
            for skill, info in in_play:
                mode = "editable" if info.editable_by_reviewer else "read-only"
                blocks.append(f"=== SKILL {skill.name} | base_hash: {skill.hash} | {mode} ===\n"
                              f"{skill.content.rstrip()}\n=== END SKILL {skill.name} ===")
            sections.append("\n\n".join(blocks))
    sections.append("## Conversation to review\n\n" + (transcript or "(empty)"))
    if focus:
        sections.append("## Focus\nThe user explicitly asked for this review. Prioritize: " + focus.strip())
    sections.append("Reply with the JSON object only.")
    return "\n\n".join(sections)


def build_repair(previous_reply: str, rejected: list) -> str:
    lines = [f"- {r.describe()}: {r.message}" for r in rejected]
    return (
        "\n\n## Your previous reply\n" + previous_reply[:8000]
        + "\n\n## Rejected operations\n" + "\n".join(lines)
        + "\n\nEverything else was applied; the memory and skills above show the current state, "
          "including fresh base_hash values. Reply with a JSON object holding corrected versions of "
          "ONLY the rejected operations, or empty lists to drop them."
    )


def render_context(user_entries: list[str], memory_entries: list[str], skills: list[tuple[str, str]],
                   *, with_tools: bool) -> str:
    """The block a host appends to its system prompt. Rendered once per session and then frozen:
    changing the system prompt mid-conversation would invalidate the provider's prompt cache."""
    if not (user_entries or memory_entries or skills or with_tools):
        return ""

    def bullets(entries):
        return "\n".join("- " + e.replace("\n", "\n  ") for e in entries)

    parts = ["# Learned context",
             "You wrote the notes below in earlier conversations. Rely on them; correct them when they are wrong."]
    if user_entries:
        parts.append("## About the user\n" + bullets(user_entries))
    if memory_entries:
        parts.append("## Your notes\n" + bullets(memory_entries))
    if skills:
        head = "## Skills\n"
        if with_tools:
            head += "Before a task that one of these covers, load it with `skill_view` and follow it.\n"
        parts.append(head + "\n".join(f"- {name}: {desc}" for name, desc in skills))
    if with_tools:
        parts.append(
            "## Saving what you learn\n"
            "Use `memory` to save a durable fact about the user or the environment. Use `skill_manage` "
            "to save or fix a reusable procedure after a non-trivial task. A reviewer also does this "
            "after the conversation, so never interrupt the user's task just to save something."
        )
    return "\n\n".join(parts)
