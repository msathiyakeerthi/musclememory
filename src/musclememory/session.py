"""One conversation's view of the learner: frozen context, tool handling, and review triggers."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING

from .prompts import render_context
from .tools import FORMATTERS, TOOL_NAMES, tool_specs
from .transcript import Turn, count_tool_rounds, count_user_turns, normalize

if TYPE_CHECKING:
    from .learner import SelfLearner

logger = logging.getLogger("musclememory")


class Session:
    """Create with :meth:`SelfLearner.session`. Typical loop::

        session = learner.session()
        system = BASE_PROMPT + "\\n\\n" + session.system_prompt()
        tools = my_tools + session.tools("anthropic")
        ...  # for every tool call the model makes:
        if session.handles(name):
            result = session.handle_tool_call(name, args)
        ...  # after the reply has been shown to the user:
        session.end_turn(messages)
        ...
        session.close()   # or use `with learner.session() as session:`
    """

    def __init__(self, learner: "SelfLearner", history=None):
        self._learner = learner
        cfg = learner.config
        lib = learner.library
        # Frozen at start: mid-session writes land on disk but not in this prompt, so the
        # provider's prompt cache stays valid for the whole conversation.
        self._user_entries = lib.memory("user")
        self._memory_entries = lib.memory("memory")
        self._skill_index = [(s.name, s.description) for s in lib.skills() if s.state != "archived"]
        self._viewed: dict[str, str] = {}  # "name" or "name/path" -> hash the agent read
        self._turns_since_memory = 0
        self._rounds_since_skill = 0
        self._wrote_memory = False
        self._wrote_skill = False
        self._consumed = 0
        self._last_turns: list[Turn] = []
        self._closed = False
        self._lock = threading.Lock()
        if history:
            turns = normalize(history)
            self._consumed, self._last_turns = len(turns), turns
            if cfg.memory_review_every_turns > 0:
                self._turns_since_memory = count_user_turns(turns) % cfg.memory_review_every_turns

    # ------------------------------------------------------------------ context + tools
    def system_prompt(self, *, with_tools: bool = True) -> str:
        """Append this to your system prompt. Identical for the life of the session."""
        return render_context(self._user_entries, self._memory_entries, self._skill_index, with_tools=with_tools)

    def tools(self, format: str = "openai") -> list[dict]:
        """Tool definitions in ``"openai"``, ``"anthropic"`` or ``"neutral"`` (JSON Schema) format."""
        if format not in FORMATTERS:
            raise ValueError(f"format must be one of {sorted(FORMATTERS)}")
        return FORMATTERS[format](tool_specs(self._learner.config.description_max_chars))

    tool_names = TOOL_NAMES

    def handles(self, name: str) -> bool:
        return name in TOOL_NAMES

    def handle_tool_call(self, name: str, arguments) -> str:
        """Run one of the learning tools. ``arguments`` may be a dict or a JSON string.
        Returns a JSON string to send back to the model as the tool result."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            return json.dumps({"success": False, "message": f"invalid arguments: {exc}"})
        handler = {"memory": self._tool_memory, "skills_list": self._tool_skills_list,
                   "skill_view": self._tool_skill_view, "skill_manage": self._tool_skill_manage}.get(name)
        if handler is None:
            return json.dumps({"success": False, "message": f"unknown tool {name!r}"})
        try:
            result = handler(args)
        except LookupError as exc:
            result = {"success": False, "message": str(exc).strip("'\"")}
        except Exception as exc:  # never let a learning tool crash the host's agent loop
            logger.exception("musclememory: tool %s failed", name)
            result = {"success": False, "message": f"internal error: {exc}"}
        return json.dumps(result, ensure_ascii=False)

    def _tool_memory(self, args: dict) -> dict:
        op = {"kind": "memory", "op": args.get("action"), "target": args.get("target"),
              "content": args.get("content"), "old_text": args.get("old_text"),
              "new_text": args.get("content")}
        result = self._learner.library.apply(op, "foreground")
        if result.ok and (result.changed or result.staged):
            self._wrote_memory = True
        return result.to_tool_result()

    def _tool_skills_list(self, args: dict) -> dict:
        skills = [{"name": s.name, "description": s.description}
                  for s in self._learner.library.skills() if s.state != "archived"]
        return {"success": True, "skills": skills}

    def _tool_skill_view(self, args: dict) -> dict:
        name, path = args.get("name"), args.get("file_path") or None
        view = self._learner.library.view_skill(name, path)
        self._viewed[name if path is None else f"{name}/{view['path']}"] = view["hash"]
        return {"success": True, **{k: v for k, v in view.items() if k != "hash"}}

    def _tool_skill_manage(self, args: dict) -> dict:
        action, name = args.get("action"), args.get("name")
        op = {"kind": "skill", "op": action, "name": name}
        if action == "create":
            op.update(description=args.get("description"), body=args.get("body"))
            key = name
        elif action == "patch":
            if name not in self._viewed:
                return {"success": False, "message": f"call skill_view({name!r}) first: edits must be based "
                                                     f"on the skill's current text"}
            op.update(old_text=args.get("old_text"), new_text=args.get("new_text"), base_hash=self._viewed[name])
            key = name
        elif action == "write_file":
            path = args.get("file_path")
            key = f"{name}/{path}"
            op.update(path=path, content=args.get("content"), base_hash=self._viewed.get(key))
        else:
            return {"success": False, "message": "action must be create, patch or write_file"}
        result = self._learner.library.apply(op, "foreground")
        if result.ok and (result.changed or result.staged):
            self._wrote_skill = True
        if result.new_hash and isinstance(key, str):
            self._viewed[key] = result.new_hash  # consecutive edits build on the agent's own write
        return result.to_tool_result()

    # ------------------------------------------------------------------ learning triggers
    def end_turn(self, messages, *, tool_rounds: int | None = None) -> Future | None:
        """Call once per user turn, AFTER the reply has been delivered, with the full message
        list so far. Returns a Future when this turn triggered a review, else None.

        ``tool_rounds`` overrides the count of tool-calling rounds detected in the new messages.
        """
        cfg = self._learner.config
        turns = normalize(messages)
        with self._lock:
            if self._closed:
                return None
            if len(turns) < self._consumed:  # a fresh list, not the cumulative one
                self._consumed = 0
            rounds = count_tool_rounds(turns[self._consumed:]) if tool_rounds is None else int(tool_rounds)
            self._consumed, self._last_turns = len(turns), turns
            # Learning in the open resets the matching counter: no need to pay for a review of
            # what the agent just saved itself.
            if not self._wrote_memory:
                self._turns_since_memory += 1
            if not self._wrote_skill:
                self._rounds_since_skill += rounds
            self._wrote_memory = self._wrote_skill = False
            memory_due = 0 < cfg.memory_review_every_turns <= self._turns_since_memory
            skill_due = 0 < cfg.skill_review_every_tool_rounds <= self._rounds_since_skill
            if not (memory_due or skill_due):
                return None
            scope = self._take_scope(memory_due, skill_due)
            viewed = self._viewed_skills()
        return self._learner._submit_review(turns, scope, viewed)

    def review_now(self, messages=None, *, scope: str = "both", focus: str | None = None) -> Future | None:
        """Force a review (like Hermes' ``/refine``); ``focus`` tells the reviewer what to look for."""
        turns = normalize(messages) if messages is not None else self._last_turns
        with self._lock:
            self._take_scope(scope in ("memory", "both"), scope in ("skills", "both"))
            viewed = self._viewed_skills()
        return self._learner._submit_review(turns, scope, viewed, focus)

    def close(self) -> Future | None:
        """End the session, reviewing whatever has not been reviewed yet (``review_on_session_end``)."""
        with self._lock:
            if self._closed:
                return None
            self._closed = True
            if not (self._learner.config.review_on_session_end and self._last_turns):
                return None
            memory_due, skill_due = self._turns_since_memory > 0, self._rounds_since_skill > 0
            if not (memory_due or skill_due):
                return None
            scope = self._take_scope(memory_due, skill_due)
            viewed = self._viewed_skills()
            turns = self._last_turns
        return self._learner._submit_review(turns, scope, viewed)

    def _take_scope(self, memory: bool, skills: bool) -> str:
        if memory:
            self._turns_since_memory = 0
        if skills:
            self._rounds_since_skill = 0
        return "both" if memory and skills else ("memory" if memory else "skills")

    def _viewed_skills(self) -> list[str]:
        return [k for k in self._viewed if "/" not in k]

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
