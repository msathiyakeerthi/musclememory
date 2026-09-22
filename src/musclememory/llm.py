"""Reviewer adapters: turn a provider client into ``callable(system, user) -> str``.

Anything with that signature works — a local model, a proxy, a test double. These helpers only
cover the common SDKs; the SDKs themselves are optional dependencies.
"""

from __future__ import annotations

from typing import Callable

LLM = Callable[[str, str], str]


class LLMRefusal(RuntimeError):
    """The model declined the review request. The review is skipped; nothing is written."""


# Server-side refusal fallbacks are a first-party Claude API feature; these clients reject it.
_NO_SERVER_FALLBACKS = ("Bedrock", "Vertex", "Foundry")


def anthropic_llm(client, model: str = "claude-opus-5", *, max_tokens: int = 16000,
                  refusal_fallbacks: bool | None = None) -> LLM:
    """Review with Claude via the official ``anthropic`` SDK.

    ``client`` is ``anthropic.Anthropic()``, or ``AnthropicBedrockMantle`` / ``AnthropicVertex`` /
    ``AnthropicFoundry`` for a cloud platform (pass that platform's model ID).

    ``refusal_fallbacks`` (default: on for the first-party API, off for cloud platforms) asks the
    API to re-run a refused request on a fallback model inside the same call.
    """
    if refusal_fallbacks is None:
        refusal_fallbacks = not any(tag in type(client).__name__ for tag in _NO_SERVER_FALLBACKS)

    def call(system: str, user: str) -> str:
        request = dict(model=model, max_tokens=max_tokens, system=system,
                       messages=[{"role": "user", "content": user}])
        if refusal_fallbacks:
            response = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
        else:
            response = client.messages.create(**request)
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"review declined ({getattr(details, 'category', None) or 'no category'})")
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")

    return call


def openai_llm(client, model: str, *, max_tokens: int | None = None, json_mode: bool = False) -> LLM:
    """Review via the ``openai`` SDK's Chat Completions API — also works with any
    OpenAI-compatible server (vLLM, Ollama, LM Studio, ...) through ``base_url``.

    ``json_mode`` sets ``response_format={"type": "json_object"}``; leave it off for servers that
    do not support it (the reply parser tolerates prose around the JSON anyway).
    """

    def call(system: str, user: str) -> str:
        request = {"model": model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if max_tokens:
            request["max_completion_tokens"] = max_tokens
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    return call
