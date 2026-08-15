"""The Anthropic Messages API provider.

Needs credit on the Console, which a Claude subscription does not include.
The reason to use it anyway is volume: a weekly run makes on the order of a
hundred generation calls that share a large identical prefix (persona, voice
profile, slop rules), and this is the only provider that can cache that
prefix. Cache reads bill at roughly a tenth of a fresh read, so on a run of
this shape it is the difference between paying for the voice profile once and
paying for it a hundred times.

That is what :class:`SystemBlock` and its ``cacheable`` flag exist for. Here
the flag becomes a real cache breakpoint; on the other providers it is
ignored.
"""

from __future__ import annotations

import json
from typing import Any

from contentsys.llm.base import (
    LLMError,
    LLMRefusal,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    Usage,
)

#: Sampling parameters are deliberately never sent. The models this system
#: targets reject them, and steering happens through the prompt and effort.


class AnthropicProvider:
    """Text completion over the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, *, api_key: str | None = None, default_model: str | None = None) -> None:
        self.default_model = default_model
        self._api_key = api_key
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Build the client lazily.

        Constructing at import time would make a missing key or a missing
        package break commands that never touch a model, such as inspecting
        config.
        """
        if self._client is not None:
            return self._client

        # The key is checked before the import on purpose. Missing credentials
        # is the far more common failure, and it is the one people misdiagnose:
        # a Claude subscription looks like it should grant API access. Leading
        # with a package install instruction would send them down the wrong
        # path, and they would still have no key at the end of it.
        if not self._api_key:
            raise LLMUnavailable(
                "no Anthropic API key. Set CONTENTSYS_ANTHROPIC_API_KEY, or switch "
                "CONTENTSYS_PROVIDER to agent_sdk to run on your Claude subscription instead. "
                "A Claude Pro subscription does not include API access; they bill separately."
            )

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise LLMUnavailable(
                'the anthropic package is not installed. Install it with: pip install -e ".[api]"'
            ) from exc

        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _system_payload(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Render system blocks, placing a cache breakpoint where marked.

        Caching is a prefix match, so a breakpoint only pays off if every
        block before it is byte identical between calls. The prompt composer
        is responsible for that ordering; this just honours the flag.
        """
        blocks: list[dict[str, Any]] = []
        for block in request.system:
            if not block.text:
                continue
            payload: dict[str, Any] = {"type": "text", "text": block.text}
            if block.cacheable:
                payload["cache_control"] = {"type": "ephemeral"}
            blocks.append(payload)
        return blocks

    def _params(self, request: LLMRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model or self.default_model or "claude-opus-5",
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if system := self._system_payload(request):
            params["system"] = system
        if request.effort:
            params["output_config"] = {"effort": request.effort}
        return params

    def complete(self, request: LLMRequest) -> LLMResponse:
        import anthropic

        try:
            response = self.client.messages.create(**self._params(request))
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable(f"the Anthropic API rejected the key: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailable(f"rate limited by the Anthropic API: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"could not reach the Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"the Anthropic API returned {exc.status_code}: {exc}") from exc

        # Check the stop reason before touching content. On a refusal the
        # content list is empty, and indexing it would raise an IndexError
        # that hides the real cause.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(
                "the model declined this request",
                category=getattr(details, "category", None),
            )

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMError(f"no text in the response (stop reason {response.stop_reason})")

        return LLMResponse(
            text=text,
            model=response.model,
            usage=_usage_from(response.usage),
            stop_reason=response.stop_reason,
            truncated=response.stop_reason == "max_tokens",
        )

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        if request.json_schema is None:
            raise ValueError("complete_json needs a request carrying a json_schema")

        params = self._params(request)
        params["output_config"] = {
            **params.get("output_config", {}),
            "format": {"type": "json_schema", "schema": request.json_schema},
        }

        import anthropic

        try:
            response = self.client.messages.create(**params)
        except anthropic.APIStatusError as exc:
            raise LLMError(f"the Anthropic API returned {exc.status_code}: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMRefusal("the model declined this request")

        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"structured output did not parse: {text[:200]}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed


def _usage_from(usage: Any) -> Usage:
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
