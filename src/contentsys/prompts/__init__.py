"""Modular prompts.

Fragments render one section each and know nothing about each other. The
composer owns ordering, including where the cache breakpoint sits.
"""

from __future__ import annotations

from contentsys.prompts.compose import (
    IDEA_SCHEMA,
    PromptContext,
    build_system,
    draft_request,
    idea_request,
)

__all__ = [
    "IDEA_SCHEMA",
    "PromptContext",
    "build_system",
    "draft_request",
    "idea_request",
]
