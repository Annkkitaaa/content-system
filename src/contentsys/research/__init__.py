"""The external layer: what happened in the world.

Kept in its own package because the boundary matters. A source says what
happened; the knowledge base says what happened to the owner. Nothing here may
become a personal claim, which is enforced by findings carrying no field an
opinion could live in, and by every idea derived from one having
``needs_experience`` forced off.
"""

from __future__ import annotations

from contentsys.research.discover import (
    CONFIDENT_DEPTHS,
    ResearchConfig,
    default_config,
    gather,
    has_standing,
    store,
    to_ideas,
)
from contentsys.research.sources import (
    Finding,
    arxiv,
    deduplicate,
    github_releases,
    hacker_news,
    rank,
    recent,
)

__all__ = [
    "CONFIDENT_DEPTHS",
    "Finding",
    "ResearchConfig",
    "arxiv",
    "deduplicate",
    "default_config",
    "gather",
    "github_releases",
    "hacker_news",
    "has_standing",
    "rank",
    "recent",
    "store",
    "to_ideas",
]
