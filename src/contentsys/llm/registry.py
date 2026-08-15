"""Choosing a provider.

One place that maps configuration to an implementation, so no call site ever
imports a concrete provider. Swapping backends is a line in .env.
"""

from __future__ import annotations

from contentsys.config import ProviderName, Settings, get_settings
from contentsys.llm.base import LLMProvider
from contentsys.llm.mock import MockProvider


def build_provider(
    name: ProviderName | str | None = None,
    settings: Settings | None = None,
) -> LLMProvider:
    """Build the configured provider.

    Concrete providers are imported inside their branch so that running on one
    backend never requires the other's package to be installed.
    """
    settings = settings or get_settings()
    resolved = ProviderName(name) if name is not None else settings.provider

    if resolved is ProviderName.MOCK:
        return MockProvider()

    if resolved is ProviderName.AGENT_SDK:
        from contentsys.llm.agent_sdk import AgentSDKProvider

        return AgentSDKProvider(default_model=settings.generation_model)

    if resolved is ProviderName.ANTHROPIC:
        from contentsys.llm.anthropic_api import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            default_model=settings.generation_model,
        )

    raise ValueError(f"unknown provider {resolved!r}")
