"""Provider adapters and framework integrations, against stand-in clients (no network)."""

import json
from types import SimpleNamespace as NS

import pytest

from musclememory import LLMRefusal, anthropic_llm, openai_llm


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def claude_response(text="{}", stop_reason="end_turn"):
    return NS(stop_reason=stop_reason, stop_details=NS(category="cyber"),
              content=[NS(type="thinking", thinking=""), NS(type="text", text=text)])


class Anthropic:  # first-party client stand-in
    def __init__(self, response):
        self.messages = FakeMessages(response)
        self.beta = NS(messages=FakeMessages(response))


class AnthropicBedrockMantle(Anthropic):  # the class name is what the adapter keys on
    pass


def test_anthropic_first_party_uses_refusal_fallbacks():
    client = Anthropic(claude_response('{"ok": true}'))
    assert anthropic_llm(client)("SYS", "USER") == '{"ok": true}'  # thinking blocks skipped
    kw = client.beta.messages.kwargs
    assert kw["model"] == "claude-opus-5" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["system"] == "SYS" and kw["messages"] == [{"role": "user", "content": "USER"}]
    assert "temperature" not in kw


def test_anthropic_cloud_platforms_skip_server_fallbacks():
    client = AnthropicBedrockMantle(claude_response())
    anthropic_llm(client, "anthropic.claude-opus-5")("s", "u")
    assert client.beta.messages.kwargs is None and client.messages.kwargs["model"] == "anthropic.claude-opus-5"


def test_anthropic_refusal_raises():
    with pytest.raises(LLMRefusal, match="cyber"):
        anthropic_llm(Anthropic(claude_response(stop_reason="refusal")))("s", "u")


def test_openai_adapter_shape():
    completions = FakeMessages(NS(choices=[NS(message=NS(content="{}"))]))
    client = NS(chat=NS(completions=completions))
    assert openai_llm(client, "some-model", json_mode=True, max_tokens=500)("SYS", "USER") == "{}"
    kw = completions.kwargs
    assert kw["messages"][0] == {"role": "system", "content": "SYS"}
    assert kw["response_format"] == {"type": "json_object"} and kw["max_completion_tokens"] == 500


def test_langchain_tools_round_trip(make_learner):
    pytest.importorskip("langchain_core")
    from musclememory.integrations.langchain import langchain_tools

    session = make_learner().session()
    tools = {t.name: t for t in langchain_tools(session)}
    assert set(tools) == set(session.tool_names)
    out = json.loads(tools["memory"].invoke({"action": "add", "target": "user", "content": "Uses LangGraph."}))
    assert out["success"]
    listed = json.loads(tools["skills_list"].invoke({}))
    assert listed == {"success": True, "skills": []}
