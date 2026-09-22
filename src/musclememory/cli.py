"""``musclememory`` — inspect and govern what an agent has learned.

    musclememory --dir ./.musclememory status
    musclememory memory | skills [--all] | show NAME [FILE]
    musclememory pending | approve ID|all | reject ID|all
    musclememory pin NAME | unpin NAME | adopt NAME | restore NAME
    musclememory ledger [-n N] | rollback ID | curate
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import LearnerConfig
from .library import Library
from .store import ENTRY_DELIMITER, FileStore


def _library(args) -> Library:
    return Library(FileStore(args.dir), LearnerConfig())


def cmd_status(lib: Library, args) -> int:
    skills = lib.skills(include_archived=True)
    by_state: dict[str, int] = {}
    for s in skills:
        by_state[s.state] = by_state.get(s.state, 0) + 1
    print(f"root: {lib.store.root}")
    for target in ("user", "memory"):
        entries = lib.memory(target)
        print(f"{target:>7} memory: {len(entries)} entries, {len(ENTRY_DELIMITER.join(entries))}/{lib.budget(target)} chars")
    print(f"  skills: {len(skills)} ({', '.join(f'{n} {k}' for k, n in sorted(by_state.items())) or 'none'})")
    print(f" pending: {len(lib.pending())}")
    print(f"  ledger: {len(lib.ledger())} changes")
    return 0


def cmd_memory(lib: Library, args) -> int:
    for target in ("user", "memory"):
        print(f"## {target}")
        for i, entry in enumerate(lib.memory(target), 1):
            print(f"[{i}] {entry}")
        print()
    return 0


def cmd_skills(lib: Library, args) -> int:
    skills = lib.skills(include_archived=args.all)
    if not skills:
        print("no skills yet")
    for s in skills:
        flags = [s.origin, s.state] + (["pinned"] if s.pinned else [])
        print(f"{s.name:<32} {s.description}\n{'':<32} [{', '.join(flags)}; used {s.use_count}x]")
    return 0


def cmd_show(lib: Library, args) -> int:
    try:
        view = lib.view_skill(args.name, args.file, record_use=False)
    except LookupError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(view["content"].rstrip())
    if view.get("files"):
        print("\nsupport files: " + ", ".join(view["files"]))
    return 0


def cmd_pending(lib: Library, args) -> int:
    items = lib.pending()
    if not items:
        print("nothing pending")
    for item in items:
        print(f"{item['id']}  [{item.get('actor')}] {item.get('summary')}  ({item.get('created_at')})")
    return 0


def _each_pending(lib: Library, ident: str) -> list[str]:
    return [i["id"] for i in lib.pending()] if ident == "all" else [ident]


def cmd_approve(lib: Library, args) -> int:
    code = 0
    for pid in _each_pending(lib, args.id):
        result = lib.approve(pid)
        print(f"{pid}: {'ok' if result.ok else 'FAILED'} — {result.message}")
        code |= 0 if result.ok else 1
    return code


def cmd_reject(lib: Library, args) -> int:
    for pid in _each_pending(lib, args.id):
        print(f"{pid}: {'rejected' if lib.reject(pid) else 'not found'}")
    return 0


def _flag(action):
    def run(lib: Library, args) -> int:
        ok = action(lib, args.name)
        print(("done: " if ok else "not found: ") + args.name)
        return 0 if ok else 1
    return run


def cmd_ledger(lib: Library, args) -> int:
    for e in lib.ledger(args.n):
        print(f"{e.get('id')}  {e.get('ts')}  {e.get('actor', '?'):<10} {e.get('action', '?'):<18} {e.get('summary', '')}")
    return 0


def cmd_rollback(lib: Library, args) -> int:
    result = lib.rollback(args.id)
    print(result.message)
    return 0 if result.ok else 1


def cmd_curate(lib: Library, args) -> int:
    report = lib.curate()
    print(f"checked {report['checked']} learned skills; stale: {report['stale'] or 'none'}; "
          f"archived: {report['archived'] or 'none'}")
    return 0


COMMANDS = {
    "status": cmd_status, "memory": cmd_memory, "skills": cmd_skills, "show": cmd_show,
    "pending": cmd_pending, "approve": cmd_approve, "reject": cmd_reject,
    "pin": _flag(lambda lib, n: lib.pin(n, True)), "unpin": _flag(lambda lib, n: lib.pin(n, False)),
    "adopt": _flag(lambda lib, n: lib.adopt(n)), "restore": _flag(lambda lib, n: lib.restore(n)),
    "ledger": cmd_ledger, "rollback": cmd_rollback, "curate": cmd_curate,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musclememory", description="Inspect and govern what an agent has learned.")
    parser.add_argument("--dir", default=os.environ.get("MUSCLEMEMORY_DIR", ".musclememory"),
                        help="learning profile directory (default: $MUSCLEMEMORY_DIR or ./.musclememory)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="counts and budgets")
    sub.add_parser("memory", help="print memory entries")
    p = sub.add_parser("skills", help="list skills")
    p.add_argument("--all", action="store_true", help="include archived skills")
    p = sub.add_parser("show", help="print a skill or one of its support files")
    p.add_argument("name")
    p.add_argument("file", nargs="?")
    sub.add_parser("pending", help="writes waiting for approval")
    for name in ("approve", "reject"):
        p = sub.add_parser(name, help=f"{name} a pending write (or 'all')")
        p.add_argument("id")
    for name, text in (("pin", "exempt from decay and autonomous edits"), ("unpin", "undo pin"),
                       ("adopt", "let the reviewer maintain a user-owned skill"), ("restore", "un-archive a skill")):
        p = sub.add_parser(name, help=text)
        p.add_argument("name")
    p = sub.add_parser("ledger", help="recent changes")
    p.add_argument("-n", type=int, default=20)
    p = sub.add_parser("rollback", help="undo one change by ledger id")
    p.add_argument("id")
    sub.add_parser("curate", help="age unused learned skills now")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](_library(args), args)


if __name__ == "__main__":
    sys.exit(main())
