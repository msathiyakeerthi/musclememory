"""The entry point: one learning profile (a directory) plus the model that reviews conversations."""

from __future__ import annotations

import atexit
import logging
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from ._util import overlap_score, tokens
from .config import LearnerConfig
from .library import Library, OpResult
from .prompts import render_context
from .review import LLM, ReviewResult, run_review
from .session import Session
from .store import FileStore
from .transcript import Turn, normalize

logger = logging.getLogger("musclememory")


class SelfLearner:
    """Make an agent self-learning.

    >>> learner = SelfLearner("./.musclememory", llm=anthropic_llm(Anthropic()))
    >>> with learner.session() as session:
    ...     system = BASE + "\\n\\n" + session.system_prompt()
    ...     ...  # your agent loop; call session.end_turn(messages) after each reply
    >>> learner.close()  # waits for in-flight reviews

    ``llm`` is any ``callable(system: str, user: str) -> str``; it only runs reviews, never your
    agent. Without it, the agent can still read and write memory/skills through the tools.
    ``on_event`` receives a dict after every review (e.g. to show "learned: ..." in a UI).
    """

    def __init__(self, root: str | Path, llm: LLM | None = None, config: LearnerConfig | None = None,
                 *, on_event: Callable[[dict], None] | None = None):
        self.config = config or LearnerConfig()
        self.store = FileStore(root)
        self.library = Library(self.store, self.config)
        self.llm = llm
        self.on_event = on_event
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future] = set()
        self._sessions: "weakref.WeakSet[Session]" = weakref.WeakSet()
        self._closed = False
        self._warned_no_llm = False
        atexit.register(self._atexit)
        if self.library.curate_due():
            self.curate()

    # ------------------------------------------------------------------ using it
    def session(self, history=None) -> Session:
        """Start a conversation. Pass ``history`` when resuming one, so the triggers pick up
        where they were instead of starting from zero."""
        session = Session(self, history)
        self._sessions.add(session)
        return session

    def system_prompt(self, *, with_tools: bool = False) -> str:
        """The learned-context block as of now — for stateless hosts that have no session."""
        lib = self.library
        skills = [(s.name, s.description) for s in lib.skills() if s.state != "archived"]
        return render_context(lib.memory("user"), lib.memory("memory"), skills, with_tools=with_tools)

    def recall(self, query: str, k: int = 2) -> str:
        """Full text of the skills that best match ``query`` — for agents without tool use,
        put this in the user turn (not the system prompt, which should stay cache-stable)."""
        q = tokens(query)
        scored = sorted(
            ((overlap_score(q, tokens(f"{s.name.replace('-', ' ')} {s.description}")), s.name)
             for s in self.library.skills() if s.state != "archived"),
            reverse=True,
        )
        blocks = []
        for score, name in scored[:k]:
            if score <= 0:
                break
            view = self.library.view_skill(name)
            blocks.append(f"## Skill: {name}\n{view['content'].strip()}")
        return "\n\n".join(blocks)

    def review(self, messages, *, scope: str = "both", focus: str | None = None) -> ReviewResult:
        """Review a conversation right now, synchronously."""
        return self._run_review(normalize(messages), scope, [], focus)

    # ------------------------------------------------------------------ background reviews
    def _submit_review(self, turns: list[Turn], scope: str, viewed: list[str],
                       focus: str | None = None) -> Future | None:
        if self.llm is None:
            if not self._warned_no_llm:
                logger.warning("musclememory: no llm configured, so background reviews are disabled")
                self._warned_no_llm = True
            return None
        if not self.config.background:
            return self._run_inline(turns, scope, viewed, focus)
        with self._lock:
            if self._closed:
                logger.warning("musclememory: learner is closed; running the review inline")
                executor = None
            else:
                if self._executor is None:
                    # One worker: reviews apply in order and never race each other's writes.
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="musclememory-review")
                executor = self._executor
            if executor is not None:
                try:
                    future = executor.submit(self._run_review, turns, scope, viewed, focus)
                except RuntimeError:  # interpreter shutting down: executors refuse new work
                    executor = None
                else:
                    self._futures.add(future)
                    future.add_done_callback(self._futures.discard)
                    return future
        return self._run_inline(turns, scope, viewed, focus)

    def _run_inline(self, turns, scope, viewed, focus) -> Future:
        future: Future = Future()
        future.set_result(self._run_review(turns, scope, viewed, focus))
        return future

    def _run_review(self, turns: list[Turn], scope: str, viewed: list[str], focus: str | None) -> ReviewResult:
        try:
            result = run_review(self.llm, self.library, turns, scope=scope, viewed=viewed, focus=focus)
        except Exception as exc:  # a failed review must never surface in the host's agent
            logger.exception("musclememory: review failed")
            result = ReviewResult(scope=scope, error=f"{type(exc).__name__}: {exc}")
        self._emit(result)
        return result

    def _emit(self, result: ReviewResult) -> None:
        if self.on_event is not None:
            try:
                self.on_event(result.to_event())
            except Exception:
                logger.exception("musclememory: on_event callback failed")
        elif result.error:
            logger.warning("musclememory: %s", result.summary())
        else:
            logger.info("musclememory: %s", result.summary())

    def wait(self, timeout: float | None = None) -> bool:
        """Block until in-flight reviews finish. True if none are left."""
        pending = list(self._futures)
        if not pending:
            return True
        _done, not_done = wait(pending, timeout=timeout)
        return not not_done

    def close(self, timeout: float | None = None) -> bool:
        """Close open sessions (running their final reviews), then wait for every review.

        Call this before your process exits — especially if anything calls ``os._exit``, which
        kills threads without waiting: a review that is not awaited is learning that never lands.
        """
        for session in list(self._sessions):
            session.close()
        finished = self.wait(self.config.close_timeout_seconds if timeout is None else timeout)
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=finished, cancel_futures=not finished)
        if not finished:
            logger.warning("musclememory: closed with reviews still running; they were abandoned")
        return finished

    def _atexit(self) -> None:
        if not self._closed:
            self.close()

    def __enter__(self) -> "SelfLearner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ governance
    def curate(self) -> dict:
        report = self.library.curate()
        if report["stale"] or report["archived"]:
            logger.info("musclememory: curator marked %d stale, archived %d", len(report["stale"]), len(report["archived"]))
        return report

    def pending(self) -> list[dict]:
        return self.library.pending()

    def approve(self, pending_id: str) -> OpResult:
        return self.library.approve(pending_id)

    def reject(self, pending_id: str) -> bool:
        return self.library.reject(pending_id)

    def rollback(self, ledger_id: str) -> OpResult:
        return self.library.rollback(ledger_id)

    def pin(self, name: str, pinned: bool = True) -> bool:
        return self.library.pin(name, pinned)

    def adopt(self, name: str) -> bool:
        return self.library.adopt(name)
