"""Twelve conversations over six sessions, then proof that session seven is better.

    python examples/multi_session_demo.py               # offline: agent and reviewer are stand-ins
    python examples/multi_session_demo.py --real        # real Claude drives both (needs an API key)

The point of the demo is the frozen-session rule: what a session learns lands on disk and shows up
in the NEXT session's system prompt. So the assistant helps with the same four kinds of task over
and over, and we count how much work each one takes before and after learning:

    preferences  -> memory   ("answer in 3 bullets", "never deploy on a Friday without asking")
    procedures   -> skills   (deploy, fix the flaky integration tests, draft release notes)

Every session gets its own SelfLearner over one shared profile directory, the way separate
processes would. Nothing is carried in memory between them; the only channel is the store.

Offline mode replaces two things and nothing else: the agent (a rule-based stand-in that reads its
own learned context) and the reviewer (a rule-based stand-in that reads the transcript). The store,
the guards, the review plumbing and the counters are the real library.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musclememory import LearnerConfig, SelfLearner  # noqa: E402
from musclememory.transcript import count_tool_rounds, normalize  # noqa: E402

# --------------------------------------------------------------------------- the conversations

# Six sessions of two conversations each. Sessions 1-3 are the discovery phase (the user states
# preferences; tasks are solved the hard way); 4-6 repeat the same classes of task.
SESSIONS: list[list[str]] = [
    ["I teach a Python class. Always answer in exactly 3 short bullets.",
     "Deploy the service to staging."],
    ["What is a list comprehension?",
     "The integration tests are failing again - can you fix them?"],
    ["Never deploy on a Friday without asking me first.",
     "Draft the release notes for v1.4."],
    ["Deploy the service to staging.",
     "What is a decorator?"],
    ["It's Friday - deploy v1.4 to production.",
     "Ok, go ahead and deploy v1.4."],
    ["Draft the release notes for v1.5.",
     "Deploy v1.5 to staging."],
]

# Between sessions 4 and 5 the release script starts refusing a deploy without VERSION set. The
# procedure the agent learned in session 1 is now wrong, which is what makes the reviewer patch it.
WORLD_CHANGE_AFTER_SESSION = 4

# The verification session: the same four classes of task, one more time, in a fresh session.
PROBES = [
    ("format", "What is a context manager?"),
    ("deploy", "Deploy the service to staging."),
    ("friday", "It's Friday - please deploy v1.5 to production."),
    ("tests", "The integration tests are failing again."),
]

TASKS = {"deploy": "deploy-python-service", "tests": "fix-flaky-integration-tests",
         "notes": "write-release-notes"}


def classify(text: str) -> str:
    low = text.lower()
    if "deploy" in low:
        return "deploy"
    if "test" in low:
        return "tests"
    if "release notes" in low:
        return "notes"
    return "explain"


# --------------------------------------------------------------------------- the fake environment

class World:
    """A tiny pretend repo. The same shell tool backs both the simulated and the real agent, so a
    run is reproducible and nothing touches the real machine."""

    def __init__(self):
        self.db_up = False
        self.requires_version = False   # flipped mid-demo: the environment moves on
        self.commands: list[str] = []   # every command of the current conversation
        self.steps: list[dict] = []     # the same commands with their output, for the trace
        self.deployed: list[str] = []

    def start_turn(self) -> None:
        self.commands, self.steps, self.deployed = [], [], []

    def shell(self, cmd: str) -> str:
        output = self._run((cmd or "").strip())
        self.steps.append({"cmd": (cmd or "").strip(), "out": output})
        return output

    def _run(self, cmd: str) -> str:
        self.commands.append(cmd)
        if cmd.startswith("npm run deploy"):
            return "error: missing script 'deploy'"
        if "make ship" in cmd:
            if self.requires_version and "VERSION=" not in cmd:
                return "error: refusing to release: VERSION not set"
            self.deployed.append(cmd)
            return "released v1.4.2"
        if cmd.startswith("cat Makefile"):
            return "ship:\n\tNODE_ENV=production ./build && ./release"
        if cmd.startswith("pytest"):
            if not self.db_up:
                return "E  psycopg.OperationalError: connection refused (is the db container up?)\n2 failed"
            return "12 passed in 4.1s"
        if cmd.startswith("docker compose ps"):
            return "NAME    STATUS\napi     running" + ("\ndb      running" if self.db_up else "")
        if cmd.startswith("docker compose up"):
            self.db_up = True
            return "Container db  Started"
        if cmd.startswith("git log --oneline"):
            return "a1b2c3d fix: retry on 502\nd4e5f6a feat: bulk export\n9f8e7d6 chore: bump deps"
        if cmd.startswith("git log --pretty=format:'- %s'"):
            return "- fix: retry on 502\n- feat: bulk export\n- chore: bump deps"
        return f"error: command not found: {cmd.split()[0] if cmd else ''}"


SHELL_TOOL = {
    "name": "shell",
    "description": "Run a shell command in the service's repository.",
    "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
}

BASE_SYSTEM = ("You are an engineering assistant for a small Python web service. Use the `shell` tool "
               "to inspect and change the repository; never guess a command's output.")


# --------------------------------------------------------------------------- the simulated agent

class SimulatedAgent:
    """Stands in for the model offline. It reads the same two inputs a real agent gets — the learned
    context in its system prompt, and the skills it can load with `skill_view` — and behaves
    differently when they hold something useful. That is what makes the before/after real: no branch
    below looks at the session number, only at what has been learned."""

    def __init__(self, world: World):
        self.world = world

    def respond(self, user_text: str, system: str, session) -> tuple[str, list[dict]]:
        task = classify(user_text)
        skill = self._load_skill(session, TASKS.get(task))
        messages: list[dict] = []

        if task == "deploy" and self._friday_rule(system) and "friday" in user_text.lower():
            return "Just to check first - you asked me never to deploy on a Friday without asking. Go ahead?", messages

        plan = self._plan(task, skill, user_text)
        while plan:
            cmd = plan.pop(0)
            messages.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{len(messages)}", "function": {"name": "shell", "arguments": json.dumps({"cmd": cmd})}}]})
            output = self.world.shell(cmd)
            messages.append({"role": "tool", "tool_call_id": f"c{len(messages)}", "content": output})
            plan = self._recover(cmd, output, user_text) + plan
        return self._reply(task, system), messages

    def _load_skill(self, session, name: str | None) -> str | None:
        """Look at the skill index, then load the body — exactly what a real agent does."""
        if not name:
            return None
        listing = json.loads(session.handle_tool_call("skills_list", {}))
        if not any(s["name"] == name for s in listing.get("skills", [])):
            return None
        view = json.loads(session.handle_tool_call("skill_view", {"name": name}))
        return view.get("content") if view.get("success") else None

    def _plan(self, task: str, skill: str | None, user_text: str) -> list[str]:
        """With the skill loaded, the agent runs the procedure the skill actually spells out — so a
        skill the reviewer later corrects changes what the agent does. Without a skill, it
        rediscovers the procedure the expensive way."""
        if skill:
            steps = re.findall(r"^\d+\. Run `(.+?)`", skill, re.M)
            if steps:
                return [self._fill_version(step, user_text) for step in steps]
        if task == "deploy":
            return ["make ship"] if skill else ["npm run deploy", "cat Makefile", "make ship"]
        if task == "tests":
            return (["docker compose up -d db", "pytest -q"] if skill
                    else ["pytest -q", "docker compose ps", "docker compose up -d db", "pytest -q"])
        if task == "notes":
            return (["git log --pretty=format:'- %s'"] if skill
                    else ["git log --oneline", "git log --pretty=format:'- %s'"])
        return []

    def _recover(self, cmd: str, output: str, user_text: str) -> list[str]:
        """What a capable agent does when a step fails: read the error and try the obvious fix. The
        reviewer then turns that recovery into a correction of the skill that misled it."""
        if "VERSION not set" in output and "VERSION=" not in cmd:
            return [self._fill_version("VERSION=<version> " + cmd, user_text)]
        return []

    @staticmethod
    def _fill_version(step: str, user_text: str) -> str:
        match = re.search(r"\bv\d+\.\d+", user_text)
        return step.replace("<version>", match.group(0) if match else "v1.4")

    @staticmethod
    def _friday_rule(system: str) -> bool:
        return "friday" in system.lower()

    def _reply(self, task: str, system: str) -> str:
        body = {"deploy": ["Released v1.4.2 to staging", "Used `make ship`", "No migrations pending"],
                "tests": ["Started the db container", "12 tests pass", "Failure was a missing database, not the code"],
                "notes": ["3 changes since v1.3", "Grouped as fix / feat / chore", "Drafted in release-notes.md"],
                "explain": ["It is a Python construct", "Used for readable, compact code", "Common in the standard library"],
                }[task]
        if "3 short bullet" in system or "three short bullet" in system:
            return "\n".join(f"- {line}" for line in body)
        return ". ".join(body) + "."


# --------------------------------------------------------------------------- the stand-in reviewer

_PREFERENCE = re.compile(r"^(?:always|never|keep|i prefer|please always)\b.*", re.I)


class HeuristicReviewer:
    """Stands in for the reviewer model offline. It reads the transcript out of the prompt the real
    `run_review` built, and returns the same JSON a model would. Two rules only: a standing
    instruction from the user becomes a memory entry, and a command that failed before one that
    worked becomes a skill (or a new pitfall on the skill that already covers that task)."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        transcript = user.split("## Conversation to review", 1)[-1]
        ops = {"memory_ops": [], "skill_ops": [], "notes": ""}
        if "memory_ops" in system or "MEMORY holds" in system:
            ops["memory_ops"] = self._memory(transcript, user)
        if "SKILLS hold" in system:
            ops["skill_ops"] = self._skills(transcript, user)
        return json.dumps(ops)

    def _memory(self, transcript: str, prompt: str) -> list[dict]:
        out = []
        for line in transcript.splitlines():
            if not line.startswith("[user] "):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line[len("[user] "):]):
                sentence = sentence.strip()
                if _PREFERENCE.match(sentence) and sentence.lower()[:40] not in prompt.split("## Skill index")[0].lower():
                    out.append({"op": "add", "target": "user", "content": sentence})
        return out

    def _skills(self, transcript: str, prompt: str) -> list[dict]:
        steps, failed = self._procedure(transcript)
        if not steps:
            return []
        # The task of the LAST user turn: an earlier turn may merely mention another kind of work
        # ("never deploy on a Friday") without being what this session actually did.
        asks = [line[len("[user] "):] for line in transcript.splitlines() if line.startswith("[user] ")]
        task = classify(asks[-1] if asks else transcript)
        name = TASKS.get(task)
        if not name:
            return []
        if not failed:
            pitfall = f"- Go straight to `{steps[0]}`; the other forms of it waste a round."
        elif failed in steps:
            pitfall = f"- `{failed}` fails until `{steps[0]}` has run first."
        else:
            pitfall = f"- `{failed}` does not work here; run `{steps[0]}` instead."
        existing = re.search(rf"=== SKILL {re.escape(name)} \| base_hash: ([0-9a-f]+) \| editable ===", prompt)
        if existing:
            # Nothing went wrong, or the skill already says so: the right answer is to save nothing.
            if not failed or pitfall in prompt:
                return []
            # Fix the step that misled, instead of appending "UPDATE:" underneath it.
            stale = re.search(rf"^(\d+\. Run `{re.escape(failed)}`\.)$", prompt, re.M)
            if stale:
                number = stale.group(1).split(".", 1)[0]
                return [{"op": "patch", "name": name, "base_hash": existing.group(1),
                         "old_text": stale.group(1),
                         "new_text": f"{number}. Run `{steps[0]}` - plain `{failed}` is refused now."}]
            return [{"op": "patch", "name": name, "base_hash": existing.group(1),
                     "old_text": "## Pitfalls", "new_text": f"## Pitfalls\n{pitfall}"}]
        description = {"deploy": "Deploy the Python service to staging or production.",
                       "tests": "Fix integration tests that fail on a missing database.",
                       "notes": "Draft release notes from the git log."}[task]
        numbered = "\n".join(f"{i}. Run `{cmd}`." for i, cmd in enumerate(steps, 1))
        body = f"{numbered}\n{len(steps) + 1}. Check the output before reporting success.\n\n## Pitfalls\n{pitfall}"
        return [{"op": "create", "name": name, "description": description, "body": body}]

    @staticmethod
    def _procedure(transcript: str) -> tuple[list[str], str | None]:
        """The signal a real reviewer reads too: what went wrong, and the run of commands that
        worked afterwards. The fix is the whole sequence — dropping the step that repaired the
        environment would leave a skill that says "run the command that already failed"."""
        ran: list[tuple[str, str]] = []
        pending: str | None = None
        for line in transcript.splitlines():
            call = re.match(r"\[assistant -> tool] shell\(\{\"cmd\": \"(.*?)\"\}\)", line)
            if call:
                pending = call.group(1)
                continue
            if line.startswith("[tool result]") and pending:
                ran.append((pending, line))
                pending = None
        broke = lambda out: "error" in out.lower() or "failed" in out.lower()     # noqa: E731
        failed = next((cmd for cmd, out in ran if broke(out)), None)
        tail: list[str] = []
        for cmd, out in reversed(ran):     # the trailing run of commands that all worked
            if broke(out):
                break
            tail.insert(0, cmd)
        steps = [c for c in tail if not c.startswith(("cat ", "ls ", "docker compose ps"))]  # diagnosis, not procedure
        if not failed and len(steps) > 1:
            steps = steps[-1:]             # nothing went wrong: the last command is the whole procedure
        return steps, failed


# --------------------------------------------------------------------------- the real-model agent

class ClaudeAgent:
    """The same loop as examples/claude_agent.py, with the shell tool wired to the fake World."""

    def __init__(self, client, model: str, world: World):
        self.client, self.model, self.world = client, model, world

    def respond(self, user_text: str, system: str, session) -> tuple[str, list[dict]]:
        messages = [{"role": "user", "content": user_text}]
        tools = [SHELL_TOOL] + session.tools("anthropic")
        while True:
            response = self.client.messages.create(model=self.model, max_tokens=4000, system=system,
                                                   tools=tools, messages=messages)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")
                return text, messages[1:]
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if session.handles(block.name):
                    output = session.handle_tool_call(block.name, block.input)
                elif block.name == "shell":
                    output = self.world.shell(block.input.get("cmd", ""))
                else:
                    output = f"error: unknown tool {block.name}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            messages.append({"role": "user", "content": results})


# --------------------------------------------------------------------------- the runner

def run_conversation(agent, session, system: str, world: World, user_text: str, history: list[dict]) -> dict:
    """One user turn, its tool work, and the bookkeeping the host is responsible for.

    ``history`` is the session's ONE growing message list: ``end_turn`` wants the whole conversation
    so far, not just this turn, or the reviewer never sees what the earlier turns revealed.
    """
    world.start_turn()
    reply, produced = agent.respond(user_text, system, session)
    turn = [{"role": "user", "content": user_text}] + produced + [{"role": "assistant", "content": reply}]
    history.extend(turn)
    session.end_turn(history)   # after the reply would have been shown
    return {"user": user_text, "reply": reply, "task": classify(user_text),
            "rounds": count_tool_rounds(normalize(turn)), "commands": list(world.commands),
            "steps": list(world.steps), "deployed": list(world.deployed)}


def run_session(make_agent, make_llm, root: Path, world: World, turns: list[str],
                verbose: bool) -> tuple[list[dict], list[str]]:
    """One session = one SelfLearner over the shared profile, the way a separate process would."""
    learned: list[str] = []
    config = LearnerConfig(background=False)  # inline reviews keep the demo's output in order
    learner = SelfLearner(root, llm=make_llm(), config=config, on_event=lambda e: learned.append(e["summary"]))
    results = []
    with learner, learner.session() as session:
        system = BASE_SYSTEM + "\n\n" + session.system_prompt()
        agent, history = make_agent(world), []
        for text in turns:
            result = run_conversation(agent, session, system, world, text, history)
            results.append(result)
            if verbose:
                print(f"    you> {text}")
                for cmd in result["commands"]:
                    print(f"      $ {cmd}")
                print("      " + result["reply"].replace("\n", "\n      "))
    for line in learned:
        print(f"    [review] {line}")
    return results, learned


def snapshot(store_root: Path) -> dict:
    """What the store holds right now — the trace's evidence that learning is just files."""
    from musclememory.library import Library
    from musclememory.store import FileStore

    lib = Library(FileStore(store_root))
    return {
        "memory": {target: lib.memory(target) for target in ("user", "memory")},
        "skills": [{"name": s.name, "description": s.description, "use_count": s.use_count,
                    "body": (lib.get_skill(s.name).content if lib.get_skill(s.name) else "")}
                   for s in lib.skills()],
        "ledger": [{"ts": e.get("ts"), "actor": e.get("actor"), "action": e.get("action"),
                    "summary": e.get("summary")} for e in lib.ledger()],
    }


def summarize(store_root: Path) -> None:
    from musclememory.library import Library
    from musclememory.store import FileStore

    lib = Library(FileStore(store_root))
    print("\n  memory (user):")
    for entry in lib.memory("user"):
        print(f"    - {entry}")
    print("  skills:")
    for info in lib.skills():
        print(f"    - {info.name}: {info.description}  [{info.origin}, used {info.use_count}x]")


def verify(make_agent, make_llm, root: Path, world: World, baseline: dict, trace: dict | None = None) -> bool:
    """A fresh session, the four task classes again, and one check per class."""
    config = LearnerConfig(background=False)
    learner = SelfLearner(root, llm=make_llm(), config=config)
    results = {}
    with learner, learner.session() as session:
        context = session.system_prompt()
        system = BASE_SYSTEM + "\n\n" + context
        print("\n  the new session's learned context:\n")
        print("    " + context.replace("\n", "\n    ").strip())
        agent, history = make_agent(world), []
        for label, text in PROBES:
            results[label] = run_conversation(agent, session, system, world, text, history)

    checks = [
        ("preference: answers in 3 bullets",
         all(line.startswith("- ") for line in results["format"]["reply"].splitlines())
         and len(results["format"]["reply"].splitlines()) == 3),
        ("preference: asks before a Friday production deploy",
         results["friday"]["deployed"] == [] and "?" in results["friday"]["reply"]),
        (f"skill: deploy takes 1 tool round and succeeds first try (was {baseline['deploy']})",
         results["deploy"]["rounds"] == 1 and results["deploy"]["deployed"] != []),
        (f"skill: flaky tests take 2 tool rounds (was {baseline['tests']})",
         results["tests"]["rounds"] == 2 and "error" not in results["tests"]["reply"].lower()),
    ]
    print()
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'XX'}] {label}")
    if trace is not None:
        trace["verification"] = {
            "context": context,
            "conversations": [dict(results[label], probe=label) for label, _text in PROBES],
            "checks": [{"label": label, "ok": ok} for label, ok in checks],
            "baseline": baseline,
        }
    return all(ok for _label, ok in checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="demo-profile", help="learning profile directory")
    parser.add_argument("--real", action="store_true", help="use Claude for the agent and the reviewer")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--quiet", action="store_true", help="only the summary and the checks")
    parser.add_argument("--trace", metavar="FILE.json", help="write the whole run as JSON (feeds the demo UI)")
    args = parser.parse_args(argv)
    getattr(sys.stdout, "reconfigure", lambda **_: None)(errors="replace")

    root = Path(args.dir)
    if root.exists():
        print(f"error: {root} already exists; delete it or pass --dir elsewhere (the demo starts from nothing)")
        return 1

    if args.real:
        import anthropic
        from musclememory import anthropic_llm
        client = anthropic.Anthropic()
        make_agent = lambda world: ClaudeAgent(client, args.model, world)          # noqa: E731
        make_llm = lambda: anthropic_llm(client, args.model)                       # noqa: E731
        print(f"=== real run: {args.model} is both the agent and the reviewer ===")
    else:
        make_agent = SimulatedAgent                                                # noqa: E731
        make_llm = HeuristicReviewer                                               # noqa: E731
        print("=== offline run: the agent and the reviewer are rule-based stand-ins ===")
        print("    (the store, guards, review pipeline and counters are the real library;")
        print("     pass --real with an ANTHROPIC_API_KEY for a run driven by Claude)")

    world = World()
    baseline: dict[str, int] = {}
    trace: dict = {"mode": "real" if args.real else "offline", "model": args.model if args.real else None,
                   "sessions": []}
    for number, turns in enumerate(SESSIONS, 1):
        print(f"\n--- session {number} ---")
        results, learned = run_session(make_agent, make_llm, root, world, turns, verbose=not args.quiet)
        for result in results:
            baseline.setdefault(result["task"], result["rounds"])
        event = None
        if number == WORLD_CHANGE_AFTER_SESSION:
            world.requires_version = True
            event = "the release script now refuses a deploy without VERSION set"
            print(f"    [world] {event}")
        trace["sessions"].append({"number": number, "conversations": results, "learned": learned,
                                  "world_event": event, "store": snapshot(root)})

    summarize(root)
    print("\n=== session 7: a new conversation, nothing carried over but the store ===")
    ok = verify(make_agent, make_llm, root, world, baseline, trace)
    if args.trace:
        Path(args.trace).write_text(json.dumps(trace, indent=1), encoding="utf-8")
        print(f"\ntrace written to {args.trace}")
    print(f"\n{'all checks passed' if ok else 'SOME CHECKS FAILED'} — profile kept at {root.resolve()}")
    print(f"  musclememory --dir {root} skills\n  musclememory --dir {root} ledger")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
