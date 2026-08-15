"""The Claude Agent SDK provider.

Runs on a Claude subscription rather than an API key, which is why it is the
default here: Claude Pro and the Anthropic API are separate products with
separate billing, and this path costs nothing beyond the subscription already
being paid for.

The SDK is Claude Code packaged as a library, so it arrives with a filesystem
agent's worth of capability that this system does not want. Every call here
is configured down to a single text in, text out exchange: no tools, no
project settings, one turn. That is deliberate. A content generator has no
business reading the filesystem, and leaving the defaults on would let a
prompt injected through a research source do exactly that.

The API shape below was verified against claude-agent-sdk 0.2.139 rather than
assumed.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from contentsys.llm.base import (
    LLMError,
    LLMRefusal,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    Usage,
)

#: Model aliases the SDK understands, so a config naming a full model id still
#: works and one naming an alias is passed through untouched.
_ALIASES = {"opus", "sonnet", "haiku"}


class AgentSDKProvider:
    """Text completion over the Claude Agent SDK."""

    name = "agent_sdk"

    def __init__(self, *, default_model: str | None = None, timeout: float = 300.0) -> None:
        self.default_model = default_model
        self.timeout = timeout

    def complete(self, request: LLMRequest) -> LLMResponse:
        return asyncio.run(self._complete_async(request))

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        if request.json_schema is None:
            raise ValueError("complete_json needs a request carrying a json_schema")
        # The SDK has no structured output parameter, so the schema goes in the
        # prompt and the response is parsed defensively. Being explicit about
        # the failure mode beats pretending the constraint is enforced.
        instructed = LLMRequest(
            system=request.system,
            prompt=(
                f"{request.prompt}\n\n"
                "Reply with JSON only, matching this schema exactly. No prose, "
                "no markdown fence, no commentary.\n"
                f"{json.dumps(request.json_schema, indent=2)}"
            ),
            max_tokens=request.max_tokens,
            model=request.model,
            effort=request.effort,
            tags=request.tags,
        )
        response = self.complete(instructed)
        return _parse_json(response.text)

    async def _complete_async(self, request: LLMRequest) -> LLMResponse:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                TextBlock,
                query,
            )
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise LLMUnavailable(
                "the claude-agent-sdk package is not installed. "
                'Install it with: pip install -e ".[agent]"'
            ) from exc

        options = ClaudeAgentOptions(
            system_prompt=request.system_text() or None,
            model=self._resolve_model(request),
            # Everything below narrows a filesystem agent to a text call.
            tools=[],
            allowed_tools=[],
            max_turns=1,
            # Do not read CLAUDE.md or project settings. This system composes
            # its own prompt deliberately, and inheriting repo instructions
            # would silently change generated voice.
            setting_sources=[],
        )

        text_parts: list[str] = []
        usage = Usage()
        stop_reason: str | None = None
        model_used = self._resolve_model(request) or "unknown"

        try:
            async with asyncio.timeout(self.timeout):
                async for message in query(prompt=request.prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        model_used = message.model or model_used
                        if message.error:
                            raise _translate_error(message.error)
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text_parts.append(block.text)
                    elif isinstance(message, ResultMessage):
                        stop_reason = message.stop_reason or message.terminal_reason
                        usage = _usage_from(message.usage)
                        if message.is_error:
                            raise LLMError(
                                "the Agent SDK reported an error: "
                                + "; ".join(message.errors or ["no detail given"])
                            )
                        break
        except TimeoutError as exc:
            raise LLMUnavailable(f"the Agent SDK did not respond within {self.timeout}s") from exc
        except (LLMError, LLMRefusal):
            raise
        except Exception as exc:
            raise LLMUnavailable(f"the Agent SDK call failed: {exc}") from exc

        text = "".join(text_parts).strip()
        if not text:
            raise LLMError("the Agent SDK returned no text")

        return LLMResponse(
            text=text,
            model=model_used,
            usage=usage,
            stop_reason=stop_reason,
            truncated=stop_reason == "max_tokens",
        )

    def _resolve_model(self, request: LLMRequest) -> str | None:
        model = request.model or self.default_model
        if model is None or model in _ALIASES:
            return model
        return model


def _translate_error(error: str) -> LLMError:
    """Turn an SDK error string into the right exception type.

    The distinction matters to callers: a refusal will fail identically on
    retry, so looping on it burns quota for nothing, while an availability
    problem is worth retrying.
    """
    if error in {"authentication_failed", "billing_error"}:
        return LLMUnavailable(
            f"the Agent SDK could not authenticate ({error}). "
            "Check that you are logged in to Claude Code, or switch "
            "CONTENTSYS_PROVIDER to anthropic and set an API key."
        )
    if "refus" in error or error == "safety":
        return LLMRefusal(f"the model declined the request ({error})", category=error)
    return LLMError(f"the Agent SDK reported {error}")


def _usage_from(raw: dict[str, Any] | None) -> Usage:
    if not raw:
        return Usage()
    return Usage(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        cache_read_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(raw.get("cache_creation_input_tokens", 0) or 0),
    )


def _parse_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a response that may be wrapped.

    Models fence JSON in markdown often enough that failing on it would be a
    self-inflicted error rate rather than a real one.
    """
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", candidate, re.S)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: the outermost brace-balanced span.
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"expected JSON, got: {text[:200]}") from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"expected JSON, got: {text[:200]}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed
