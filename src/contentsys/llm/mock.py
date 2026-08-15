"""A deterministic, offline provider.

This exists so the entire pipeline, including a full weekly run of 72 pieces,
is testable without a network call, an API key, or a cent of spend. That is
not a convenience: a pipeline you cannot afford to run in CI is a pipeline
nobody refactors safely.

Determinism comes from seeding on the request itself, so the same prompt
always yields the same output and a test can assert on it. The generated text
is deliberately written in the target voice (lowercase, short, no em dashes)
so that pipeline tests exercise realistic input rather than lorem ipsum that
would sail through every evaluator.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from contentsys.llm.base import LLMError, LLMRequest, LLMResponse, Usage

#: Fragments in the owner's register. Short, lowercase, concrete.
_OPENERS = (
    "i used to think",
    "one thing i keep coming back to is",
    "the part nobody mentions about",
    "spent a while confused by",
    "still not over how",
)

_SUBJECTS = (
    "sumcheck",
    "the reduction from r1cs to a polynomial",
    "polynomial commitments",
    "how spartan avoids a trusted setup",
    "why the verifier only checks one point",
)

_TURNS = (
    "turns out the whole trick is that you never check the thing directly.",
    "it clicks the moment you read it as a chain of reductions, not a protocol.",
    "the elegance is that none of the pieces are clever on their own.",
    "the interesting bit is what you are allowed to not compute.",
    "you are not proving the value, you are proving something satisfying it exists.",
)


class MockProvider:
    """Offline provider with stable, voice-plausible output."""

    name = "mock"

    def __init__(self, *, fail_on: str | None = None) -> None:
        #: When set, any prompt containing this substring raises. Lets tests
        #: exercise the regeneration and error paths deliberately.
        self.fail_on = fail_on
        self.calls: list[LLMRequest] = []

    def _rng(self, request: LLMRequest) -> random.Random:
        material = f"{request.system_text()}|{request.prompt}|{request.model}"
        seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
        return random.Random(seed)

    def _record(self, request: LLMRequest) -> None:
        self.calls.append(request)
        if self.fail_on and self.fail_on in request.prompt:
            raise LLMError(f"mock provider was configured to fail on {self.fail_on!r}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._record(request)
        rng = self._rng(request)
        text = f"{rng.choice(_OPENERS)} {rng.choice(_SUBJECTS)}. {rng.choice(_TURNS)}"
        return LLMResponse(
            text=text,
            model=request.model or "mock",
            usage=self._usage(request, len(text)),
            stop_reason="end_turn",
        )

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        self._record(request)
        if request.json_schema is None:
            raise ValueError("complete_json needs a request carrying a json_schema")
        rng = self._rng(request)
        return _synthesise(request.json_schema, rng)

    def _usage(self, request: LLMRequest, output_chars: int) -> Usage:
        # Roughly four characters per token. Good enough to make accounting
        # code exercise real arithmetic rather than zeros.
        prompt_chars = len(request.system_text()) + len(request.prompt)
        cacheable = sum(len(b.text) for b in request.system if b.cacheable)
        return Usage(
            input_tokens=max(1, (prompt_chars - cacheable) // 4),
            output_tokens=max(1, output_chars // 4),
            cache_read_tokens=cacheable // 4,
        )


def _synthesise(schema: dict[str, Any], rng: random.Random) -> Any:
    """Build a value that satisfies a JSON Schema.

    Covers the subset the prompts actually use: objects with properties,
    arrays, strings with enums, numbers with bounds, booleans. Anything
    unrecognised falls back to a string, which surfaces as a validation error
    downstream rather than silently passing.
    """
    kind = schema.get("type", "string")

    if kind == "object":
        properties: dict[str, Any] = schema.get("properties", {})
        required = schema.get("required", list(properties))
        return {name: _synthesise(properties[name], rng) for name in required if name in properties}

    if kind == "array":
        item_schema = schema.get("items", {"type": "string"})
        lower = schema.get("minItems", 1)
        upper = schema.get("maxItems", max(lower, 3))
        return [_synthesise(item_schema, rng) for _ in range(rng.randint(lower, upper))]

    if kind == "boolean":
        return rng.choice([True, False])

    if kind in {"number", "integer"}:
        low = schema.get("minimum", 0)
        high = schema.get("maximum", 10)
        value = rng.uniform(low, high)
        return int(value) if kind == "integer" else round(value, 1)

    if enum := schema.get("enum"):
        return rng.choice(enum)

    rng_choice = f"{rng.choice(_OPENERS)} {rng.choice(_SUBJECTS)}"
    return rng_choice


def canonical_request_key(request: LLMRequest) -> str:
    """A stable key for a request. Used by tests that assert on call shape."""
    return hashlib.sha256(
        json.dumps(
            {
                "system": [b.text for b in request.system],
                "prompt": request.prompt,
                "model": request.model,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
