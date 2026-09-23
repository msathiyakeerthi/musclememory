"""Build the demo pages from a run's trace.

    python examples/assistant_demo.py --dir /tmp/profile --trace docs/assistant-trace.json
    python docs/build_ui.py

`assistant-ui.html` is rendered here, in full: every chat and every note is plain HTML in the file,
so the page shows the whole story at once and needs no scripting to display it. `demo-ui.html` is
the interactive one for engineers; it only gets its `const TRACE = ...;` line refreshed.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


# --------------------------------------------------------------------------- assistant page

def bubbles(messages: list[dict]) -> str:
    out = []
    for m in messages:
        if m["role"] == "action":
            out.append(f'<div class="act"><span class="dot" aria-hidden="true"></span>'
                       f'<span>{esc(m["text"])} <span class="detail">&mdash; {esc(m.get("detail", ""))}</span>'
                       f"</span></div>")
            continue
        who = "them" if m["role"] == "person" else "ai"
        flag = " correction" if m.get("correction") else ""
        label = "Person" if m["role"] == "person" else "Assistant"
        out.append(f'<div class="bubble {who}{flag}"><span class="sr">{label}: </span>{esc(m["text"])}</div>')
    return "\n".join(out)


def chat_block(chat: dict) -> str:
    tags = []
    if chat["questions"]:
        tags.append(f'<span class="tag">had to ask {chat["questions"]} question'
                    f'{"s" if chat["questions"] > 1 else ""}</span>')
    if chat["corrections"]:
        tags.append('<span class="tag warn">got it wrong</span>')
    if not tags:
        tags.append('<span class="tag">no questions, no corrections</span>')
    return (f'<div class="chat"><p class="clock">{esc(chat["clock"])}</p>\n{bubbles(chat["messages"])}\n'
            f'<div class="tags">{"".join(tags)}</div></div>')


def skill_note(skill: dict, kind: str) -> str:
    steps = re.findall(r"^\d+\. (.+)$", skill["body"], re.M)
    matters = re.findall(r"^- (.+)$", skill["body"], re.M)
    steps_html = "<ol>" + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ol>" if steps else ""
    matters_html = "".join(f'<div class="matters">&bull; {esc(m)}</div>' for m in matters)
    return (f'<div class="note skl"><span class="kind">{kind}</span>'
            f'<span class="title">{esc(skill["description"])}</span>{steps_html}{matters_html}</div>')


def wrote_block(day: dict, previous: dict | None) -> str:
    """What this day added to the store, compared with the day before."""
    before_memory = set(previous["memory"]) if previous else set()
    before_skills = {s["name"]: s["body"] for s in (previous["skills"] if previous else [])}
    new_memory = [e for e in day["store"]["memory"] if e not in before_memory]
    new_skills = [(s, "a new thing it knows how to do") for s in day["store"]["skills"]
                  if s["name"] not in before_skills]
    fixed_skills = [(s, "corrected what it already knew") for s in day["store"]["skills"]
                    if s["name"] in before_skills and before_skills[s["name"]] != s["body"]]

    if not (new_memory or new_skills or fixed_skills):
        return ('<div class="wrote quiet"><div class="head"><h3>After these chats</h3></div>'
                '<p class="what">Nothing new to write down &mdash; it already knew everything these '
                "chats needed. Most days end like this.</p></div>")

    notes = "".join(f'<div class="note mem"><span class="kind">remembers about you</span>{esc(e)}</div>'
                    for e in new_memory)
    notes += "".join(skill_note(s, kind) for s, kind in new_skills + fixed_skills)
    return ('<div class="wrote"><div class="head"><h3>After these chats it wrote</h3></div>'
            f'{notes}</div>')


def tally(chat: dict, win: bool) -> str:
    klass = ' class="win"' if win else ""
    return (f'<div{klass}><b>{chat["questions"]}</b> question{"" if chat["questions"] == 1 else "s"} it had to ask'
            f'<br><b>{chat["corrections"]}</b> time{"" if chat["corrections"] == 1 else "s"} it got it wrong</div>')


def assistant_page(trace: dict) -> str:
    first, now = trace["finale"]["first"], trace["finale"]["now"]
    parts = [
        '<p class="kicker">Muscle Memory</p>',
        "<h1>The assistant that stops asking you the same thing</h1>",
        '<p class="lede">Twelve chats with one household over six days. Nothing is configured and no model is '
        "retrained. After each chat, the assistant writes down what it should have known &mdash; and the next "
        "chat starts with those notes in hand. Here is the whole run, in order.</p>",
        '<div class="result">',
        f'<div><p class="when">Day one &mdash; &ldquo;{esc(first["messages"][0]["text"])}&rdquo;</p>'
        f'<p class="line">It asked <b>{first["questions"]}</b> questions before it could start, then booked '
        "the table over the school run.</p></div>",
        f'<div class="win"><p class="when">Two weeks later &mdash; the same request</p>'
        f'<p class="line"><b>{now["questions"]}</b> questions, <b>{now["corrections"]}</b> mistakes. '
        "It checked the calendar, knew the household is vegetarian, and asked before booking.</p></div>",
        "</div>",
    ]

    previous = None
    for day in trace["days"]:
        parts.append('<section class="day">')
        parts.append(f'<header><h2>Day {day["number"]}</h2><span class="num">{esc(day["label"])}</span></header>')
        parts.extend(chat_block(chat) for chat in day["chats"])
        parts.append(wrote_block(day, previous))
        parts.append("</section>")
        previous = day["store"]

    parts += [
        '<section class="finale">',
        "<h2>The same request, two weeks apart</h2>",
        '<p class="lede">Nobody repeated a single instruction in between.</p>',
        '<div class="grid">',
        f'<div class="side"><h3>Day one</h3>{bubbles(first["messages"])}'
        f'<div class="foot">{tally(first, False)}</div></div>',
        f'<div class="side"><h3>Two weeks later</h3>{bubbles(now["messages"])}'
        f'<div class="foot">{tally(now, True)}</div></div>',
        "</div></section>",
        '<div class="how">',
        "<div><b>1. It just has the conversation</b><p>Nothing special happens while you are talking. The "
        "reply is never held up by learning.</p></div>",
        "<div><b>2. A reviewer reads it afterwards</b><p>A second pass over the finished chat asks one "
        "question: is anything here worth knowing next time? Usually the answer is no.</p></div>",
        "<div><b>3. The next chat starts with it</b><p>What it wrote is a short text file a person can read, "
        "edit or delete. It goes into the next conversation.</p></div>",
        "</div>",
        "<footer><p>A real run of the open-source library, rendered here: the notes above are the exact files "
        "it produced. The assistant and the reviewer are rule-based stand-ins so the demo runs without an API "
        "key &mdash; the memory, the skills, the safety rules and the review pipeline are the real thing. "
        '<a href="https://github.com/msathiyakeerthi/musclememory">github.com/msathiyakeerthi/musclememory</a>'
        "</p></footer>",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- build

def build_assistant() -> None:
    trace = json.loads((HERE / "assistant-trace.json").read_text(encoding="utf-8"))
    template = (HERE / "assistant-ui.template.html").read_text(encoding="utf-8")
    # A plain string replace, never re.sub: backslashes in the content are literal, not escapes.
    page = template.replace("<!--CONTENT-->", assistant_page(trace))
    (HERE / "assistant-ui.html").write_text(page, encoding="utf-8")
    print(f"assistant-ui.html: {round(len(page.encode()) / 1024)} KB, {len(trace['days'])} days")


def build_trace_page(page_name: str, trace_name: str) -> None:
    page, trace = HERE / page_name, HERE / trace_name
    if not (page.exists() and trace.exists()):
        print(f"skipped {page_name}: missing page or trace")
        return
    data = json.dumps(json.loads(trace.read_text(encoding="utf-8")), separators=(",", ":"))
    text = page.read_text(encoding="utf-8")
    match = re.search(r"const TRACE = .*?;\n", text, re.S)
    if not match:
        print(f"skipped {page_name}: no `const TRACE = ...;` line")
        return
    page.write_text(text[: match.start()] + f"const TRACE = {data};\n" + text[match.end():], encoding="utf-8")
    print(f"{page_name}: {round(page.stat().st_size / 1024)} KB")


def main() -> int:
    build_assistant()
    build_trace_page("demo-ui.html", "demo-trace.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
