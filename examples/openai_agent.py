"""The same learning agent on the OpenAI Chat Completions API (or any OpenAI-compatible server).

    pip install "musclememory[openai]"
    python examples/openai_agent.py --model <model-id>
    python examples/openai_agent.py --model <model-id> --base-url http://localhost:11434/v1   # e.g. Ollama
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAI

from musclememory import SelfLearner, openai_llm

sys.path.insert(0, __file__.rsplit("examples", 1)[0] + "examples")
from claude_agent import BASE_SYSTEM, CALCULATE_TOOL, calculate  # noqa: E402  (shared demo tool)

CALCULATE = {"type": "function", "function": {"name": CALCULATE_TOOL["name"],
                                              "description": CALCULATE_TOOL["description"],
                                              "parameters": CALCULATE_TOOL["input_schema"]}}


def run_turn(client, model: str, tools: list, messages: list, session) -> str:
    while True:
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content or ""
        for call in message.tool_calls:
            name, raw_args = call.function.name, call.function.arguments
            if session.handles(name):                                  # <- musclememory
                output = session.handle_tool_call(name, raw_args)
            elif name == "calculate":
                output = calculate(json.loads(raw_args or "{}").get("expression", ""))
            else:
                output = f"error: unknown tool {name}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint")
    parser.add_argument("--dir", default=".musclememory")
    parser.add_argument("--say", action="append")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
    learner = SelfLearner(args.dir, llm=openai_llm(client, args.model),
                          on_event=lambda e: print(f"  [musclememory] {e['summary']}", flush=True))
    with learner, learner.session() as session:
        messages: list = [{"role": "system", "content": BASE_SYSTEM + "\n\n" + session.system_prompt()}]
        tools = [CALCULATE] + session.tools("openai")
        turns = iter(args.say or [])
        while True:
            text = next(turns, None) if args.say else input("you> ").strip()
            if not text or text in ("/quit", "/exit"):
                break
            messages.append({"role": "user", "content": text})
            print(f"assistant> {run_turn(client, args.model, tools, messages, session)}\n", flush=True)
            session.end_turn(messages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
