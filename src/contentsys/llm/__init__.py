"""Model access.

Concrete providers arrive in the generation phase. Everything before that
codes against the protocol in :mod:`contentsys.llm.base`.
"""

from __future__ import annotations

from contentsys.llm.base import (
    LLMError,
    LLMProvider,
    LLMRefusal,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    SystemBlock,
    Usage,
    collect_usage,
    system_prompt,
)

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMRefusal",
    "LLMRequest",
    "LLMResponse",
    "LLMUnavailable",
    "SystemBlock",
    "Usage",
    "collect_usage",
    "system_prompt",
]
