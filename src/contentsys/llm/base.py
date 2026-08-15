"""The LLM boundary.

Every part of the system that needs a model talks to :class:`LLMProvider` and
nothing else. Three implementations land in a later phase:

``AgentSDKProvider``
    Runs on a Claude subscription through the Claude Agent SDK. No API key.
``AnthropicProvider``
    The Messages API. Needs credit on the Console, supports prompt caching
    and batching, which matter for a 70-post weekly run.
``MockProvider``
    Deterministic and offline, so the whole pipeline is testable for free.

The request shape is deliberately narrow. It carries what every backend can
honour and nothing that only one of them understands, so swapping providers
never means rewriting call sites.

One shape choice worth explaining: ``system`` is a sequence of blocks rather
than a single string. The stable part of a prompt (persona, voice profile,
slop rules) is identical across roughly a hundred calls in a weekly run, so
marking it ``cacheable`` lets the Messages API provider place a cache
breakpoint there. Backends that cannot cache simply concatenate and ignore
the flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMRefusal",
    "LLMRequest",
    "LLMResponse",
    "LLMUnavailable",
    "SystemBlock",
    "Usage",
    "system_prompt",
]


class LLMError(RuntimeError):
    """Base class for anything that went wrong at the model boundary."""


class LLMUnavailable(LLMError):
    """The backend could not be reached, or is not configured.

    Raised for a missing API key, a missing subscription login, an exhausted
    credit, or a transport failure that survived the provider's own retries.
    """


class LLMRefusal(LLMError):
    """The model declined the request.

    Distinct from a transport failure: retrying the identical prompt will
    fail the same way, so callers should rephrase, route elsewhere, or drop
    the item rather than loop.
    """

    def __init__(self, message: str, *, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True, slots=True)
class SystemBlock:
    """One span of system prompt.

    Set ``cacheable`` on the last block of the stable prefix. Because caching
    is a prefix match, a block marked cacheable is only useful if every block
    before it is byte-identical across calls, so put volatile content last
    and never interpolate a timestamp into an early block.
    """

    text: str
    cacheable: bool = False


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One completion request.

    ``temperature`` is deliberately absent. The models this system targets
    reject sampling parameters, and steering happens through the prompt and
    the effort level instead.
    """

    system: tuple[SystemBlock, ...]
    prompt: str
    max_tokens: int = 4096
    model: str | None = None
    effort: str | None = None

    #: When set, the provider constrains output to this JSON Schema and
    #: :meth:`LLMProvider.complete_json` returns the parsed object.
    json_schema: dict[str, Any] | None = None

    #: Free-form labels carried through to usage records. Used to attribute
    #: spend to a stage (ideas, drafting, one named evaluator) so a weekly
    #: run can be costed per component rather than as one opaque number.
    tags: tuple[str, ...] = ()

    def system_text(self) -> str:
        """The system prompt as one string, for backends without blocks."""
        return "\n\n".join(block.text for block in self.system if block.text)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for a single call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def billable_input_tokens(self) -> int:
        """Input tokens that were not served from cache.

        Cache reads are roughly a tenth the price of a fresh read, so this is
        the number to watch when tuning where the cache breakpoint sits.
        """
        return self.input_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The result of one completion."""

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None

    #: True when the response was cut short by the token ceiling. Callers that
    #: care about complete output should check this rather than trusting that
    #: a non-empty string means a finished thought.
    truncated: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    """What every backend must offer.

    Implementations are expected to handle their own retries for transient
    failures and to raise :class:`LLMUnavailable` only once retrying is
    pointless. A refusal must raise :class:`LLMRefusal` rather than returning
    an empty or apologetic string, so the pipeline never scores a refusal as
    if it were a draft.
    """

    #: Stable identifier, matching a ``ProviderName`` value.
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion and return the raw text."""
        ...

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        """Run one completion constrained to ``request.json_schema``.

        Raises :class:`ValueError` if the request carries no schema, and
        :class:`LLMError` if the backend returned something that does not
        parse or does not validate.
        """
        ...


def system_prompt(
    *parts: str | SystemBlock, cache_through: int | None = None
) -> tuple[SystemBlock, ...]:
    """Build a system prompt from ordered fragments.

    Plain strings become non-cacheable blocks. ``cache_through`` marks the
    block at that index as the cache breakpoint, which is the last block of
    the stable prefix. Negative indices work the way they do in a list.

    Empty fragments are dropped, so a caller can pass an optional section
    without branching around it.
    """
    blocks = [
        part if isinstance(part, SystemBlock) else SystemBlock(part)
        for part in parts
        if (part.text if isinstance(part, SystemBlock) else part)
    ]
    if cache_through is not None and blocks:
        index = cache_through if cache_through >= 0 else len(blocks) + cache_through
        if not 0 <= index < len(blocks):
            raise IndexError(
                f"cache_through {cache_through} is out of range for {len(blocks)} blocks"
            )
        blocks[index] = SystemBlock(blocks[index].text, cacheable=True)
    return tuple(blocks)


def collect_usage(responses: Sequence[LLMResponse]) -> Usage:
    """Sum token usage across a batch of calls."""
    total = Usage()
    for response in responses:
        total = total + response.usage
    return total
