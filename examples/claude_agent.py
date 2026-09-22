"""A complete Claude tool-use agent that learns across conversations.

    pip install "musclememory[anthropic]"            # or "musclememory[bedrock]"
    python examples/claude_agent.py                # interactive chat
    python examples/claude_agent.py --say "..." --say "..."    # scripted turns
    python examples/claude_agent.py --bedrock us-west-2        # via Amazon Bedrock

Run it twice. In the first run, tell it about yourself or correct how it answers; the second
run starts already knowing. Inspect what it learned with `musclememory --dir .musclememory memory`.
"""

from __future__ import annotations

import argparse
import ast
import operator
import sys

import anthropic

from musclememory import SelfLearner, anthropic_llm

BASE_SYSTEM = "You are a helpful assistant. Use the `calculate` tool for arithmetic instead of doing it in your head."

CALCULATE_TOOL = {
    "name": "calculate",
    "description": "Evaluate an arithmetic expression such as '17*23 + 4'. Numbers and + - * / ( ) only.",
    "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> str:
    """Arithmetic via the AST — never eval() model output."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ( ) are allowed")
    try:
        return str(ev(ast.parse(expression, mode="eval")))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"error: {exc}"


def make_create(client, first_party: bool):
    """messages.create, with server-side refusal fallbacks where the platform supports them."""
    if first_party:
        return lambda **kw: client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
    return client.messages.create


def run_turn(create, model: str, system: str, tools: list, messages: list, session) -> str:
    """The standard manual tool loop; the only musclememory-specific lines route its tools."""
    while True:
        response = create(model=model, max_tokens=16000, system=system, tools=tools, messages=messages)
        messages.append({"role": "assistant", "content": response.content})  # keep thinking blocks intact
        if response.stop_reason == "refusal":
            return "(the model declined this request)"
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if session.handles(block.name):                      # <- musclememory
                output = session.handle_tool_call(block.name, block.input)
            elif block.name == "calculate":
                output = calculate(block.input.get("expression", ""))
            else:
                output = f"error: unknown tool {block.name}"
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})  # all results in ONE message


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".musclememory", help="learning profile directory")
    parser.add_argument("--model", help="agent model (default: claude-opus-5; on Bedrock anthropic.claude-opus-5)")
    parser.add_argument("--reviewer-model", help="model for background reviews (default: same as --model)")
    parser.add_argument("--bedrock", metavar="REGION", help="use Amazon Bedrock in this AWS region")
    parser.add_argument("--bedrock-runtime", action="store_true",
                        help="with --bedrock: use the bedrock-runtime (InvokeModel) client and an inference-profile "
                             "model ID, for accounts where the Messages-API (Mantle) endpoint is not available")
    parser.add_argument("--say", action="append", help="scripted user turn (repeatable); omit for interactive chat")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if args.bedrock and args.bedrock_runtime:
        client = anthropic.AnthropicBedrock(aws_region=args.bedrock)
        model = args.model or "us.anthropic.claude-opus-5"  # on-demand calls need an inference profile
    elif args.bedrock:
        client = anthropic.AnthropicBedrockMantle(aws_region=args.bedrock)
        model = args.model or "anthropic.claude-opus-5"
    else:
        client = anthropic.Anthropic()
        model = args.model or "claude-opus-5"
    create = make_create(client, first_party=not args.bedrock)

    def on_event(event: dict) -> None:
        print(f"  [musclememory] {event['summary']}", flush=True)
        if event.get("notes"):
            print(f"  [musclememory] reviewer notes: {event['notes']}", flush=True)

    learner = SelfLearner(args.dir, llm=anthropic_llm(client, args.reviewer_model or model), on_event=on_event)
    with learner, learner.session() as session:  # leaving: final review, then wait for it
        system = BASE_SYSTEM + "\n\n" + session.system_prompt()
        tools = [CALCULATE_TOOL] + session.tools("anthropic")
        messages: list = []
        turns = iter(args.say) if args.say else None
        while True:
            if turns is not None:
                text = next(turns, None)
                if text is None:
                    break
                print(f"you> {text}")
            else:
                try:
                    text = input("you> ").strip()
                except EOFError:
                    break
                if text in ("", "/quit", "/exit"):
                    break
            messages.append({"role": "user", "content": text})
            print(f"assistant> {run_turn(create, model, system, tools, messages, session)}\n", flush=True)
            session.end_turn(messages)  # after the reply is shown — learning never delays it
        print("[musclememory] reviewing the conversation before exit…", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
