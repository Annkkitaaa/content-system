"""Turning external facts into things worth writing about.

The boundary this module defends: a finding says what happened, an idea says
what the owner makes of it, and the two must never merge. Every idea produced
here carries its source and is forced to ``needs_experience = False``, because
nothing the world did is something the owner did.

It also filters on standing. A story the owner has no basis to comment on is
dropped rather than written up: reacting to a paper in a field you have never
read is how an account starts sounding like it is performing expertise.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlmodel import Session, select

from contentsys.config import CONFIG_DIR, Platform
from contentsys.content.ideas import Idea
from contentsys.db.models import KnowledgeItem, ResearchSource
from contentsys.research.sources import (
    Finding,
    arxiv,
    deduplicate,
    github_releases,
    hacker_news,
    rank,
    recent,
)

#: Depths that give the owner standing to comment publicly. Someone who has
#: only heard of a topic should not be reacting to news about it.
CONFIDENT_DEPTHS = frozenset({"deep", "working", "familiar"})

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class ResearchConfig:
    reactive_share: float = 0.25
    max_age_days: int = 10
    per_source_limit: int = 12
    hacker_news: dict = None  # type: ignore[assignment]
    arxiv: dict = None  # type: ignore[assignment]
    github_releases: dict = None  # type: ignore[assignment]
    avoid: list[str] = None  # type: ignore[assignment]

    @classmethod
    def load(cls, path: Path | None = None) -> ResearchConfig:
        source = path or (CONFIG_DIR / "research.yaml")
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        return cls(
            reactive_share=float(data.get("reactive_share", 0.25)),
            max_age_days=int(data.get("max_age_days", 10)),
            per_source_limit=int(data.get("per_source_limit", 12)),
            hacker_news=data.get("hacker_news") or {},
            arxiv=data.get("arxiv") or {},
            github_releases=data.get("github_releases") or {},
            avoid=[str(item).lower() for item in (data.get("avoid") or [])],
        )

    def reactive_count(self, total: int) -> int:
        """How many of this week's posts should react to something recent."""
        return max(0, min(total, round(total * self.reactive_share)))


@functools.lru_cache(maxsize=1)
def default_config() -> ResearchConfig:
    return ResearchConfig.load()


def gather(config: ResearchConfig | None = None) -> list[Finding]:
    """Fetch, filter, deduplicate and rank everything worth knowing about."""
    config = config or default_config()
    findings: list[Finding] = []

    if config.hacker_news.get("enabled", True):
        findings.extend(
            hacker_news(
                config.hacker_news.get("queries", []),
                limit=config.per_source_limit,
                min_points=int(config.hacker_news.get("min_points", 5)),
            )
        )

    if config.arxiv.get("enabled", True):
        findings.extend(
            arxiv(
                config.arxiv.get("categories", ["cs.CR"]),
                config.arxiv.get("keywords", []),
                limit=config.per_source_limit,
            )
        )

    if config.github_releases.get("enabled", True):
        findings.extend(
            github_releases(
                config.github_releases.get("repositories", []),
                limit=config.per_source_limit,
            )
        )

    findings = recent(findings, max_age_days=config.max_age_days)
    findings = [f for f in findings if not _avoided(f, config.avoid)]
    return rank(deduplicate(findings))


def _avoided(finding: Finding, avoid: list[str]) -> bool:
    lowered = finding.title.lower()
    return any(term in lowered for term in avoid)


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if len(word) > 3}


def has_standing(finding: Finding, knowledge: list[KnowledgeItem]) -> bool:
    """Whether the owner can credibly comment on this.

    Reacting to a paper in a field you have never read is how an account
    starts performing expertise instead of having it, and no amount of careful
    phrasing downstream fixes that.
    """
    words = _terms(f"{finding.title} {finding.summary}")
    for item in knowledge:
        if item.depth not in CONFIDENT_DEPTHS:
            continue
        concept = _terms(item.concept)
        if concept and concept <= words:
            return True
        if any(term in words for term in concept if len(term) > 4):
            return True
    return False


def to_ideas(
    findings: list[Finding],
    platform: Platform,
    *,
    content_type: str = "reaction",
) -> list[Idea]:
    """Turn findings into ideas, keeping the source visible.

    ``needs_experience`` is forced off. Nothing the world did is something the
    owner did, and an idea that asked for a first-person claim here would be
    asking the generator to invent one.
    """
    ideas: list[Idea] = []
    for finding in findings:
        context = f"Source: {finding.source}. {finding.title}"
        if finding.summary:
            context += f"\n{finding.summary[:400]}"
        ideas.append(
            Idea(
                topic=finding.title[:90],
                angle=(
                    "React to this as external news, not as something you did. State "
                    "what happened, then what you make of it, keeping the two apart. "
                    "Only claim what you can actually support.\n\n" + context
                ),
                why_interesting=f"recent, from {finding.source}",
                content_type=content_type,
                platform=platform,
                source=finding.url or finding.source,
                needs_experience=False,
                novelty=min(10.0, 6.0 + max(0.0, 4.0 - finding.age_days)),
            )
        )
    return ideas


def store(session: Session, findings: list[Finding]) -> int:
    """Record findings so the workbook can cite them.

    Matched on URL so a re-run does not duplicate the same story, and so the
    Research sheet stays a record rather than a growing pile.
    """
    added = 0
    for finding in findings:
        if finding.url:
            existing = session.exec(
                select(ResearchSource).where(ResearchSource.url == finding.url)
            ).first()
            if existing is not None:
                continue
        session.add(
            ResearchSource(
                title=finding.title,
                url=finding.url,
                kind=finding.source.lower().replace(" ", "_"),
                key_fact=finding.summary[:400] or finding.title,
                topic=finding.tags[0] if finding.tags else None,
            )
        )
        added += 1
    return added
