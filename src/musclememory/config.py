"""Tuning knobs. Defaults follow the Hermes Agent learning loop; set a trigger to 0 to disable it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LearnerConfig:
    # --- when to review -----------------------------------------------------------------
    memory_review_every_turns: int = 10
    """Review for memory after this many user turns. A chatty session reveals facts."""

    skill_review_every_tool_rounds: int = 10
    """Review for skills after this many tool-calling rounds. Hard work is what discovers a procedure."""

    review_on_session_end: bool = True
    """Review unreviewed activity when a session closes. Without it, sessions shorter than the
    intervals above never learn anything."""

    background: bool = True
    """Run reviews on a worker thread so they never delay the user's reply."""

    # --- what may be written ------------------------------------------------------------
    write_approval: bool = False
    """Stage every agent-initiated write for human approval instead of applying it."""

    memory_char_budget: int = 2200
    user_char_budget: int = 1400
    description_max_chars: int = 60
    skill_body_max_chars: int = 20_000
    support_file_max_chars: int = 50_000

    # --- the reviewer -------------------------------------------------------------------
    review_transcript_max_chars: int = 60_000
    review_max_skill_bodies: int = 4
    review_repair_rounds: int = 1
    """After rejected operations, show the reviewer the errors and let it correct them this many times."""

    extra_review_instructions: str = ""
    """Domain rules appended to the reviewer prompt, e.g. "Never store customer names or order IDs."."""

    # --- decay --------------------------------------------------------------------------
    stale_after_days: int = 14
    archive_after_days: int = 30
    curate_every_days: float = 7.0

    # --- shutdown -----------------------------------------------------------------------
    close_timeout_seconds: float = 180.0
    """How long ``close()`` (and interpreter exit) waits for in-flight reviews."""
