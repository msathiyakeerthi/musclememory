"""Any agent's message format reads into the same turns."""

from types import SimpleNamespace as NS

from musclememory.transcript import count_tool_rounds, count_user_turns, normalize, render


def test_openai_chat_format():
    turns = normalize([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "deploy it"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function", "function": {"name": "shell", "arguments": '{"cmd": "make ship"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
        {"role": "assistant", "content": "Deployed."},
    ])
    assert [t.role for t in turns] == ["system", "user", "assistant", "tool", "assistant"]
    assert turns[2].tool_calls == [("shell", '{"cmd": "make ship"}')]
    assert count_tool_rounds(turns) == 1 and count_user_turns(turns) == 1


def test_anthropic_blocks_as_dicts_and_sdk_objects():
    turns = normalize([
        {"role": "user", "content": "deploy it"},
        {"role": "assistant", "content": [
            NS(type="thinking", thinking="hmm"),
            NS(type="text", text="Running it."),
            NS(type="tool_use", id="t1", name="shell", input={"cmd": "make ship"}),
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "ok"}]}]},
    ])
    assert turns[1].text == "Running it." and turns[1].tool_calls == [("shell", '{"cmd": "make ship"}')]
    assert turns[2].role == "tool" and turns[2].text == "ok"
    assert count_user_turns(turns) == 1  # tool results are not user turns


def test_langchain_style_objects():
    turns = normalize([
        NS(type="human", content="hi"),
        NS(type="ai", content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "1"}]),
        NS(type="tool", content="found"),
    ])
    assert [t.role for t in turns] == ["user", "assistant", "tool"]
    assert turns[1].tool_calls == [("search", '{"q": "x"}')]


def test_render_keeps_the_tail():
    turns = normalize([{"role": "user", "content": f"message {i} " + "x" * 50} for i in range(100)])
    text = render(turns, 500)
    assert text.startswith("[…") and "message 99" in text and "message 0 " not in text
    assert "[system]" not in render(normalize([{"role": "system", "content": "s"}]), 100)
