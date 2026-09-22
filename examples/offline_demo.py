"""Watch the whole loop with no API key — a scripted stand-in plays the reviewer.

    python examples/offline_demo.py

Useful for teaching: every step the real system takes is printed, including the guard that
refuses a skill description over 60 characters and the reviewer fixing it.
"""

from __future__ import annotations

import json
import sys
import tempfile

from musclememory import SelfLearner

LONG = "A comprehensive skill that deploys the Python service to production using the make ship target."


def scripted_reviewer():
    replies = iter([
        # 1st reply: a memory fact, plus a skill whose description is too long (it will be refused)
        json.dumps({
            "memory_ops": [{"op": "add", "target": "user",
                            "content": "Teaches a class; wants answers as three short bullet points."}],
            "skill_ops": [{"op": "create", "name": "deploy-python-service", "description": LONG,
                           "body": "1. Run `make ship`, never `npm run deploy`.\n\n## Pitfalls\n"
                                   "- Set NODE_ENV=production first - the build silently ships a dev bundle otherwise."}],
        }),
        # repair round: the reviewer is shown the error and shortens the description
        json.dumps({"memory_ops": [], "skill_ops": [{
            "op": "create", "name": "deploy-python-service", "description": "Deploy the Python service to production.",
            "body": "1. Run `make ship`, never `npm run deploy`.\n\n## Pitfalls\n"
                    "- Set NODE_ENV=production first - the build silently ships a dev bundle otherwise."}]}),
    ])

    def reviewer(system: str, user: str) -> str:
        print(f"\n  [reviewer called: {len(system)} chars of rules, {len(user)} chars of context]")
        if "Rejected operations" in user:
            print("  [reviewer was shown its rejected operation and is correcting it]")
        return next(replies, '{"memory_ops": [], "skill_ops": []}')

    return reviewer


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    root = tempfile.mkdtemp(prefix="musclememory-demo-")
    learner = SelfLearner(root, llm=scripted_reviewer(), on_event=lambda e: (
        print(f"  [event] {e['summary']}"),
        [print(f"  [rejected] {r['op']}: {r['error'][:90]}…") for r in e["rejected"]],
    ))

    print("=== Session 1: the agent knows nothing yet ===")
    with learner.session() as session:
        print(session.system_prompt() or "(no learned context)")
        def shell(i, cmd, out):
            return [{"role": "assistant", "content": None,
                     "tool_calls": [{"id": str(i), "function": {"name": "shell", "arguments": json.dumps({"cmd": cmd})}}]},
                    {"role": "tool", "tool_call_id": str(i), "content": out}]

        messages = (
            [{"role": "user", "content": "I teach a class. Deploy the service - and keep answers to 3 bullets."}]
            + shell(1, "npm run deploy", "error: missing script 'deploy'")
            + shell(2, "cat Makefile", "ship:\n\tNODE_ENV=production ./build && ./release")
            + shell(3, "make ship", "released v1.4.2")
            + [{"role": "assistant", "content": "- Deployed v1.4.2\n- Used `make ship`\n- npm has no deploy script"}]
        )
        future = session.end_turn(messages)  # 1 turn, 3 tool rounds: below both intervals (10)...
        print(f"\nend_turn -> {future}  (no review yet: below the intervals)")
    learner.wait()                            # ...but closing the session reviewed it

    print("\n=== Session 2: a new conversation starts already knowing ===")
    print(learner.session().system_prompt())
    learner.close()
    print(f"\nStore kept at {root}. Inspect it:\n"
          f"  musclememory --dir \"{root}\" skills\n  musclememory --dir \"{root}\" ledger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
