"""A personal assistant that stops asking the same questions — twelve chats over six days.

    python examples/assistant_demo.py --trace docs/assistant-trace.json
    python examples/assistant_demo.py --real            # Claude plays the assistant and the reviewer

Same machinery as examples/multi_session_demo.py, but the story is one anybody recognises: an
assistant that books restaurants, orders groceries and plans trips. On day one it has to ask what
you eat, and it books over your calendar. Nobody tells it the rules twice — after each chat a
reviewer writes down what it should have known, and the next chat starts with that in hand.

What gets written is exactly the two stores:
    memory  - facts about the person ("the household is vegetarian", "no peanuts - allergy")
    skills  - how to do a recurring job ("book a restaurant: check the calendar first, confirm
              before booking")

Offline, the assistant and the reviewer are rule-based stand-ins; the store, the guards, the review
pipeline and the counters are the real library. Nothing in the assistant looks at the day number:
it behaves differently only because it reads what earlier days wrote.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musclememory import LearnerConfig, SelfLearner  # noqa: E402

# --------------------------------------------------------------------------- the chats

# Each chat: what the person opens with, plus the answers they give if the assistant has to ask.
# `correction` is what the person says after the assistant gets something wrong.
CHATS = [
    # ---- day 1
    {"day": "Tuesday", "task": "restaurant", "clock": "09:14",
     "open": "Book us a table for four this Friday at 7.",
     "answers": {"diet": "We're all vegetarian, so somewhere with proper vegetarian mains.",
                 "budget": "Around 30 a head, nothing fancy."},
     "correction": "You booked over the school run - please always check my calendar before booking anything."},
    {"day": "Tuesday", "task": "groceries", "clock": "18:40",
     "open": "Can you put in the weekly grocery order?",
     "answers": {"shop": "Same shop as always, the usual weekly list.",
                 "diet": "Nothing with meat in it."},
     "correction": "Please take the peanut butter off - my son is allergic to peanuts, so no peanuts ever."},

    # ---- day 2
    {"day": "Thursday", "task": "restaurant", "clock": "11:02",
     "open": "Dinner for my parents on Saturday, same sort of place.",
     "answers": {},
     "correction": "Next time check with me before you actually book - I'd have picked the later slot."},
    {"day": "Thursday", "task": "trip", "clock": "20:15",
     "open": "Plan us a weekend by the coast next month.",
     "answers": {"budget": "Keep it under 600 for the three of us.",
                 "travel": "We'd rather take the train than drive."},
     "correction": None},

    # ---- day 3
    {"day": "Monday", "task": "groceries", "clock": "08:05",
     "open": "Weekly shop, please.", "answers": {}, "correction": None},
    {"day": "Monday", "task": "restaurant", "clock": "17:30",
     "open": "Table for two on Wednesday evening?", "answers": {}, "correction": None},

    # ---- day 4
    {"day": "Wednesday", "task": "trip", "clock": "09:47",
     "open": "Another weekend away, this time in the hills.", "answers": {}, "correction": None},
    {"day": "Wednesday", "task": "groceries", "clock": "19:22",
     "open": "Same grocery order as last week.", "answers": {}, "correction": None},

    # ---- day 5
    {"day": "Friday", "task": "restaurant", "clock": "10:10",
     "open": "Lunch spot for my book club on Sunday?", "answers": {}, "correction": None},
    {"day": "Friday", "task": "groceries", "clock": "16:55",
     "open": "Groceries again please.", "answers": {}, "correction": None},

    # ---- day 6
    {"day": "Tuesday", "task": "trip", "clock": "12:30",
     "open": "Half term is coming - somewhere by the sea again.", "answers": {}, "correction": None},
    {"day": "Tuesday", "task": "restaurant", "clock": "21:05",
     "open": "Anniversary dinner next Thursday, something a bit nicer.", "answers": {}, "correction": None},
]

CHATS_PER_DAY = 2

# The chat the whole demo is built to answer: the very first request, asked again at the end.
FINALE = {"day": "the following week", "task": "restaurant", "clock": "09:14",
          "open": "Book us a table for four this Friday at 7.", "answers": {}, "correction": None}

SKILLS = {"restaurant": "book-a-restaurant", "groceries": "weekly-grocery-order", "trip": "plan-a-weekend-trip"}

# What the assistant needs to know before it can act without asking. Each fact is remembered as a
# memory entry; the phrase is what the stand-in reviewer looks for in the person's own words.
FACTS = {
    "diet": ("vegetarian", "The household is vegetarian - no meat or fish, ever."),
    "allergy": ("peanut", "No peanuts in anything - the son has a peanut allergy."),
    "budget": ("a head", "Eating out: around 30 a head unless told otherwise."),
    "travel": ("train", "Prefers the train over driving for trips."),
}


# --------------------------------------------------------------------------- the pretend world

class World:
    """A calendar with one clash, and three little services the assistant can call."""

    CLASH = "Friday 19:00 - school run"

    def act(self, name: str, detail: str = "") -> str:
        if name == "check_calendar":
            return f"Clash: {self.CLASH}. Free from 20:00." if "friday" in detail.lower() else "Nothing booked."
        if name == "search_restaurants":
            veg = "vegetarian" in detail.lower()
            return ("The Gate - vegetarian, 28 a head, 2 tables left" if veg
                    else "Smokehouse - steak, 45 a head, tables all evening")
        if name == "book_table":
            return f"Booked: {detail}"
        if name == "order_groceries":
            return f"Basket ready: {detail}"
        if name == "search_trips":
            return "Whitstable by train, 2 nights, 540 total for three"
        return "done"


STEP_LABEL = {                # how the same action reads as a step in a written procedure
    "check_calendar": "Check the calendar for clashes",
    "search_restaurants": "Find a place that fits their diet and budget",
    "book_table": "Book the table",
    "order_groceries": "Fill the basket from the usual weekly list",
    "search_trips": "Find trips that match the budget and how they like to travel",
}

ACTION_LABEL = {              # what the person sees the assistant doing, in the chat
    "check_calendar": "Checked the calendar",
    "search_restaurants": "Looked for restaurants",
    "book_table": "Booked the table",
    "order_groceries": "Filled the basket",
    "search_trips": "Looked up trips",
}


# --------------------------------------------------------------------------- the assistant

class SimulatedAssistant:
    """Its whole personality is three questions: what do I already know about this person, what
    procedure have I written for this job, and what do I still have to ask?"""

    def __init__(self, world: World):
        self.world = world

    def chat(self, chat: dict, context: str, session) -> list[dict]:
        """Returns the chat as a list of {role, text} — role is person, assistant or action."""
        known = {key: phrase in context.lower() for key, (phrase, _entry) in FACTS.items()}
        skill = self.load_skill(session, SKILLS[chat["task"]])
        out: list[dict] = [{"role": "person", "text": chat["open"]}]

        for key, question in self.questions(chat["task"]):
            if known.get(key) or key not in chat["answers"]:
                continue
            out.append({"role": "assistant", "text": question, "question": True})
            out.append({"role": "person", "text": chat["answers"][key]})
            known[key] = True

        for step in self.steps(chat["task"], skill, known):
            out.append({"role": "action", "text": ACTION_LABEL[step["do"]],
                        "detail": self.world.act(step["do"], step["detail"])})
            if step.get("then_confirm"):
                out.append({"role": "assistant", "text": step["then_confirm"], "confirm": True})

        if not any(m.get("confirm") for m in out):
            out.append({"role": "assistant", "text": self.closing(chat["task"], skill, known)})
        if chat["correction"]:
            out.append({"role": "person", "text": chat["correction"], "correction": True})
            out.append({"role": "assistant", "text": "Sorry about that - I'll remember."})
        return out

    @staticmethod
    def load_skill(session, name: str) -> str | None:
        listing = json.loads(session.handle_tool_call("skills_list", {}))
        if not any(s["name"] == name for s in listing.get("skills", [])):
            return None
        view = json.loads(session.handle_tool_call("skill_view", {"name": name}))
        return view.get("content") if view.get("success") else None

    @staticmethod
    def questions(task: str) -> list[tuple[str, str]]:
        if task == "restaurant":
            return [("diet", "Happy to. Any dietary preferences I should work around?"),
                    ("budget", "And roughly what budget per person?")]
        if task == "groceries":
            return [("diet", "Sure - anything the household doesn't eat?")]
        return [("budget", "Of course. What sort of budget are we working with?"),
                ("travel", "And would you rather drive or take the train?")]

    def steps(self, task: str, skill: str | None, known: dict) -> list[dict]:
        """With a skill, the assistant follows the procedure it wrote down — including the steps a
        correction added. Without one, it does the obvious thing and sometimes gets it wrong."""
        learned = (skill or "").lower()
        if task == "restaurant":
            plan: list[dict] = []
            if "calendar" in learned:
                plan.append({"do": "check_calendar", "detail": "Friday"})
            plan.append({"do": "search_restaurants", "detail": "vegetarian" if known.get("diet") else ""})
            if "confirm" in learned:
                # It found a table but stops short of booking: the correction taught it to ask first.
                plan[-1]["then_confirm"] = ("The Gate has 20:00 free - vegetarian, about 28 a head. "
                                            "Shall I book that?")
            else:
                plan.append({"do": "book_table", "detail": "The Gate, Friday 19:00, four people"})
            return plan
        if task == "groceries":
            return [{"do": "order_groceries",
                     "detail": "the usual weekly list" + (", no meat" if known.get("diet") else "")
                               + (", nothing with peanuts" if known.get("allergy") else "")}]
        return [{"do": "search_trips", "detail": "train" if known.get("travel") else "any"}]

    @staticmethod
    def skill_steps(skill: str | None) -> str:
        return (skill or "").lower()

    @staticmethod
    def closing(task: str, skill: str | None, known: dict) -> str:
        if task == "restaurant":
            if skill and "confirm" in skill.lower():
                return "Nothing is booked until you say go."
            return "Booked - I've put it in your calendar."
        if task == "groceries":
            extras = []
            if known.get("diet"):
                extras.append("all vegetarian")
            if known.get("allergy"):
                extras.append("no peanuts")
            return "Basket is ready" + (f" ({', '.join(extras)})" if extras else "") + " - shall I place it?"
        return "Whitstable by train, two nights, 540 for the three of you. Want me to hold it?"


# --------------------------------------------------------------------------- the stand-in reviewer

class HeuristicReviewer:
    """Reads the transcript out of the real review prompt and returns the JSON a model would.
    Two rules: something the person states about themselves becomes memory; a correction after the
    assistant acted becomes (or fixes) the skill for that job."""

    def __call__(self, system: str, user: str) -> str:
        transcript = user.split("## Conversation to review", 1)[-1]
        said = "\n".join(line for line in transcript.splitlines() if line.startswith("[user] "))
        ops = {"memory_ops": [], "skill_ops": [], "notes": ""}
        if "MEMORY holds" in system:
            for key, (phrase, entry) in FACTS.items():
                if phrase in said.lower() and entry.lower()[:30] not in user.split("## Skill index")[0].lower():
                    ops["memory_ops"].append({"op": "add", "target": "user", "content": entry})
        if "SKILLS hold" in system:
            ops["skill_ops"] = self._skill(transcript, said, user)
        return json.dumps(ops)

    DESCRIPTIONS = {"restaurant": "Book a restaurant the way this household likes it.",
                    "groceries": "Place the household's weekly grocery order.",
                    "trip": "Plan a weekend away for the family."}

    JOB_WORDS = (("restaurant", ("table", "dinner", "lunch", "restaurant")),
                 ("groceries", ("grocer", "shop", "basket")),
                 ("trip", ("weekend", "trip", "coast", "hills", "sea")))

    def _skill(self, transcript: str, said: str, prompt: str) -> list[dict]:
        """A session can cover several jobs, so each job gets its own operation."""
        ops = []
        for task, (did, spoken) in self._by_job(transcript).items():
            name = SKILLS[task]
            lesson = self._lesson(spoken)
            existing = re.search(rf"=== SKILL {re.escape(name)} \| base_hash: ([0-9a-f]+) \| editable ===", prompt)
            if existing:
                if lesson and lesson not in prompt:
                    ops.append({"op": "patch", "name": name, "base_hash": existing.group(1),
                                "old_text": "## What matters", "new_text": f"## What matters\n- {lesson}"})
                continue
            if not did or re.search(rf"^- {re.escape(name)}:", prompt, re.M):
                continue      # no actions to describe, or the skill exists but wasn't shown in full
            steps = "\n".join(f"{i}. {STEP_LABEL[a]}." for i, a in enumerate(did, 1))
            matters = f"- {lesson}" if lesson else "- Say what was booked or ordered, in one line."
            ops.append({"op": "create", "name": name, "description": self.DESCRIPTIONS[task],
                        "body": f"{steps}\n\n## What matters\n{matters}"})
        return ops

    def _by_job(self, transcript: str) -> dict:
        """Walk the session in order, following which job each stretch of it is about, and collect
        that job's actions and the words the person said about it."""
        jobs: dict = {}
        current = None
        for line in transcript.splitlines():
            if line.startswith("[user] "):
                low = line.lower()
                for task, keywords in self.JOB_WORDS:
                    if any(k in low for k in keywords):
                        current = task
                        break
                if current:
                    jobs.setdefault(current, ([], []))[1].append(line)
            elif current and line.startswith("[assistant -> tool] "):
                action = line[len("[assistant -> tool] "):].split("(")[0].strip()
                actions = jobs.setdefault(current, ([], []))[0]
                if action in ACTION_LABEL and action not in actions:
                    actions.append(action)
        return jobs

    @staticmethod
    def _lesson(said) -> str | None:
        low = "\n".join(said).lower() if isinstance(said, list) else said.lower()
        if "check my calendar" in low:
            return "Check the calendar before booking - evenings often have the school run."
        if "check with me before" in low:
            return "Offer the option and wait for a yes; never book without confirming."
        if "allergic" in low:
            return "Never order anything with peanuts - it is an allergy, not a preference."
        return None


# --------------------------------------------------------------------------- the runner

BASE_SYSTEM = ("You are a personal assistant for a busy parent. Use your tools to check the calendar, "
               "find places and place orders. Keep replies to a sentence or two.")


def to_messages(chat_messages: list[dict]) -> list[dict]:
    """The chat as the model-facing message list the library reads: an action becomes a tool call
    and its result, which is what makes the tool-round counter tick."""
    out: list[dict] = []
    for i, m in enumerate(chat_messages):
        if m["role"] == "person":
            out.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            out.append({"role": "assistant", "content": m["text"]})
        else:
            call = {"id": f"a{i}", "function": {"name": m["do"], "arguments": json.dumps({"detail": m.get("q", "")})}}
            out.append({"role": "assistant", "content": None, "tool_calls": [call]})
            out.append({"role": "tool", "tool_call_id": f"a{i}", "content": m["detail"]})
    return out


def score(chat_messages: list[dict]) -> dict:
    """How much the person had to do: questions they answered, and times they had to correct it."""
    return {"questions": sum(1 for m in chat_messages if m.get("question")),
            "corrections": sum(1 for m in chat_messages if m.get("correction")),
            "exchanges": len(chat_messages)}


def run_day(assistant_for, llm_for, root: Path, chats: list[dict], quiet: bool) -> tuple[list[dict], list[str]]:
    """One day = one session = one SelfLearner over the shared profile, as a separate process would."""
    learned: list[str] = []
    def note(event: dict) -> None:
        learned.append(event["summary"])
        for rejection in event["rejected"]:
            print(f"    [refused] {rejection['op']}: {rejection['error'][:110]}")

    learner = SelfLearner(root, llm=llm_for(), config=LearnerConfig(background=False), on_event=note)
    done = []
    with learner, learner.session() as session:
        context = session.system_prompt()
        assistant, history = assistant_for(World()), []
        for chat in chats:
            messages = assistant.chat(chat, context, session)
            for m in messages:
                if m["role"] == "action":
                    m["do"] = next(k for k, v in ACTION_LABEL.items() if v == m["text"])
            history.extend(to_messages(messages))
            session.end_turn(history)
            done.append({"day": chat["day"], "clock": chat["clock"], "task": chat["task"],
                         "messages": messages, **score(messages)})
            if not quiet:
                print(f"    {chat['clock']}  {chat['open']}")
                for m in messages[1:]:
                    mark = {"person": "them", "assistant": "  AI", "action": "   ·"}[m["role"]]
                    print(f"      {mark}  {m['text']}")
    for line in learned:
        print(f"    [review] {line}")
    return done, learned


def store_snapshot(root: Path) -> dict:
    from musclememory.library import Library
    from musclememory.store import FileStore

    lib = Library(FileStore(root))
    return {"memory": lib.memory("user"),
            "skills": [{"name": s.name, "description": s.description,
                        "body": (lib.get_skill(s.name).content if lib.get_skill(s.name) else "")}
                       for s in lib.skills()]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="assistant-profile")
    parser.add_argument("--trace", metavar="FILE.json", help="write the whole run as JSON (feeds the demo UI)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    getattr(sys.stdout, "reconfigure", lambda **_: None)(errors="replace")

    root = Path(args.dir)
    if root.exists():
        print(f"error: {root} already exists; delete it or pass --dir elsewhere (the demo starts from nothing)")
        return 1

    trace: dict = {"mode": "offline", "days": []}
    for index in range(0, len(CHATS), CHATS_PER_DAY):
        chats = CHATS[index:index + CHATS_PER_DAY]
        day_number = index // CHATS_PER_DAY + 1
        print(f"\n--- day {day_number} ({chats[0]['day']}) ---")
        done, learned = run_day(SimulatedAssistant, HeuristicReviewer, root, chats, args.quiet)
        trace["days"].append({"number": day_number, "label": chats[0]["day"], "chats": done,
                              "learned": learned, "store": store_snapshot(root)})

    print("\n--- the same request as day 1, asked again ---")
    finale, _learned = run_day(SimulatedAssistant, HeuristicReviewer, root, [FINALE], args.quiet)
    first = trace["days"][0]["chats"][0]
    last = finale[0]
    trace["finale"] = {"first": first, "now": last, "store": store_snapshot(root)}

    ok = last["questions"] == 0 and last["corrections"] == 0 and first["questions"] > 0
    print(f"\n  day 1:  {first['questions']} questions asked, {first['corrections']} correction(s)")
    print(f"  now:    {last['questions']} questions asked, {last['corrections']} correction(s)")
    print(f"\n{'the assistant needed nothing repeated' if ok else 'CHECK FAILED'} — profile at {root.resolve()}")
    if args.trace:
        Path(args.trace).write_text(json.dumps(trace, indent=1), encoding="utf-8")
        print(f"trace written to {args.trace}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
