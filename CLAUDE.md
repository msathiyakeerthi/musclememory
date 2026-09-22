# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`musclememory` is a Python library (>=3.10) that adds a self-learning loop (long-term memory + on-demand skills + a background LLM reviewer) to any LLM agent. The design is distilled from Hermes Agent's learning loop. The core package is **stdlib-only** (`dependencies = []` in `pyproject.toml`). Provider SDKs (`anthropic`, `openai`, `langchain-core`) are optional extras and must only be imported lazily or inside adapters and examples, never at core module import time.

## Commands

A virtualenv lives at `.venv/` (Windows: `.venv/Scripts/python.exe`).

```bash
pip install -e ".[dev,anthropic,openai,langchain]"   # dev setup
pytest                                               # full suite, offline (68 tests, ~3s)
pytest tests/test_library.py                         # one file
pytest tests/test_session.py::test_name              # one test
pytest -k rollback                                   # by keyword
python examples/offline_demo.py                      # whole loop end to end, no API key
musclememory --dir .musclememory status                    # CLI (entry point: musclememory.cli:main)
```

No linter or formatter is configured. Match the existing style: `from __future__ import annotations`, long lines (~120), and dense docstrings that explain *why*.

## Architecture

Full detail is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The short version:

The public API is re-exported from `src/musclememory/__init__.py`. The layers, top to bottom:

- **`SelfLearner`** (`learner.py`): one learning profile, meaning a root directory plus the reviewer `llm`, which is any `callable(system: str, user: str) -> str`. It owns a **single-worker** `ThreadPoolExecutor`, so reviews apply in order and never race. `close()` closes the tracked sessions (a `WeakSet`), which runs their final reviews, then waits. An `atexit` hook does the same on exit. It also runs the weekly curator (decay) at construction when due.
- **`Session`** (`session.py`): one conversation. The learned context is **frozen at construction** so the host's prompt cache stays valid; mid-session writes appear only in the next session. It exposes the 4 learning tools, and `handle_tool_call` always returns a JSON string and never raises into the host loop. `end_turn(messages)` counts user turns (which trigger memory reviews) and tool-calling rounds (which trigger skill reviews). A counter resets when the agent saved that kind of thing itself during the turn. The session tracks `_viewed` content hashes: `skill_manage patch` is refused unless `skill_view` was called first.
- **Reviewer** (`review.py` + `prompts.py`): one LLM call per review that must return a JSON object `{memory_ops, skill_ops, notes}`. `extract_json` tolerates code fences and prose. If the reply fails to parse, it retries once. Rejected ops are shown back to the reviewer for `review_repair_rounds` rounds. The reviewer's behavioural rules live in the `prompts.py` text (class-level skills, patch before create, a do-not-capture list). Changing that text changes learning quality.
- **`Library`** (`library.py`): the **single choke point for every write**. `apply(op, actor)` is called by the agent's tools (`"foreground"`), the reviewer (`"review"`) and the CLI (`"user"`). It enforces sanitize/scan (`guards.py`: secrets, prompt-injection phrases, invisible/bidi Unicode), the 60-char description cap (refused, never truncated), base-hash checks, char budgets, and ownership. Skill `origin` (`learned`/`agent`/`user`) decides editability: the reviewer may only edit `learned`, unpinned skills. With `write_approval`, non-user writes are validated with `dry_run` and staged under `state/pending/`. Every commit is appended to the ledger with its prior content, which is what enables `rollback`. `curate()` moves skills `active → stale → archived` and never deletes them.
- **`FileStore`** (`store.py`): plain files under the root. `memory/USER.md` and `memory/MEMORY.md` hold entries separated by a line containing only `§`. Each skill is `skills/<name>/SKILL.md` (flat frontmatter) plus optional `references/`, `templates/`, `scripts/`. The `state/` directory holds `usage.json`, `ledger.jsonl`, `pending/` and `curator.json`. Writes are atomic (temp file + `os.replace`, with Windows retry) and **always LF**, because the `§` delimiter is newline-sensitive.
- **`transcript.py`**: `normalize()` reads OpenAI, Anthropic (dicts or SDK blocks) and LangChain message lists, freely mixed, into `Turn`s. All counting and rendering go through it, so support for a new message format belongs here.
- **Adapters**: `llm.py` (`anthropic_llm`, `openai_llm`; `LLMRefusal`), `tools.py` (neutral JSON Schema specs plus `openai`/`anthropic` formatters), `integrations/langchain.py`.

Invariant: failures in learning must never break the host agent. Reviews, tool calls, `on_event` callbacks and ops all catch exceptions and log them to the `musclememory` logger instead of raising. Keep that property when adding code paths.

## Tests

Tests are fully offline. `tests/conftest.py` provides `ScriptedLLM`, a fake reviewer that returns queued replies (strings, or callables of `(system, user)`) and records every prompt in `.calls`. It also provides `ops(memory=..., skills=...)` to build reviewer JSON, a `lib` fixture (a bare `Library` on `tmp_path`), and a `make_learner(*replies, **config)` factory that closes the learners on teardown.
