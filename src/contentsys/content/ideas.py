"""The idea engine.

Ideas are generated before drafts, and separately, for one reason: generating
70 posts from 70 ideas means writing up 70 mediocre ideas. Oversampling the
pool and dropping the weakest is the only cheap way to raise the floor.

An idea is a topic plus the specific angle that makes it worth writing. A
subject area is not an idea. "Sumcheck" is not an idea. "Sumcheck is the only
part of the stack that got simpler as it got more general" is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from contentsys.config import Platform
from contentsys.llm.base import LLMProvider
from contentsys.prompts import PromptContext, idea_request

#: Two angles this similar are the same idea wearing different words.
_DUPLICATE_THRESHOLD = 0.62

_WORD = re.compile(r"[a-z0-9']+")

_FILLER = frozenset(
    """
    a an and are as at be but by for from how in into is it its of on or that the
    their them then there these they this to was what when which who why with you
    your i me my we our
    """.split()
)


@dataclass
class Idea:
    """A candidate, before anything is written."""

    topic: str
    angle: str
    why_interesting: str
    content_type: str
    platform: Platform
    technical_depth: str = "medium"
    novelty: float = 5.0
    needs_experience: bool = False
    experience_id: int | None = None
    source: str | None = None
    rejected_because: str | None = None

    @property
    def usable(self) -> bool:
        return self.rejected_because is None

    def fingerprint(self) -> frozenset[str]:
        """Content words from the angle, which is where the idea actually lives.

        Topic is deliberately excluded. Ten good ideas about sumcheck should
        not collapse into one; ten rephrasings of the same claim should.
        """
        words = _WORD.findall(f"{self.angle} {self.why_interesting}".lower())
        return frozenset(word for word in words if word not in _FILLER and len(word) > 2)


@dataclass
class IdeaPool:
    """Generated ideas, with the record of what was dropped and why."""

    ideas: list[Idea] = field(default_factory=list)
    dropped: list[Idea] = field(default_factory=list)

    @property
    def usable(self) -> list[Idea]:
        return [idea for idea in self.ideas if idea.usable]

    def summary(self) -> str:
        parts = [f"{len(self.usable)} usable"]
        if self.dropped:
            reasons: dict[str, int] = {}
            for idea in self.dropped:
                key = idea.rejected_because or "unknown"
                reasons[key] = reasons.get(key, 0) + 1
            parts.append(
                "dropped: " + ", ".join(f"{n} {reason}" for reason, n in sorted(reasons.items()))
            )
        return "; ".join(parts)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Overlap between two word sets, 0 to 1."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def deduplicate(ideas: Iterable[Idea], threshold: float = _DUPLICATE_THRESHOLD) -> IdeaPool:
    """Drop ideas that restate one already kept.

    Compares angles rather than topics, and keeps the first occurrence. Higher
    novelty ideas are expected to have been sorted to the front already, so
    first-wins means best-wins.
    """
    pool = IdeaPool()
    kept_fingerprints: list[tuple[frozenset[str], Idea]] = []

    for idea in ideas:
        fingerprint = idea.fingerprint()
        clash = next(
            (
                other
                for existing, other in kept_fingerprints
                if jaccard(fingerprint, existing) >= threshold
            ),
            None,
        )
        if clash is not None:
            idea.rejected_because = "duplicate angle"
            pool.dropped.append(idea)
            continue
        kept_fingerprints.append((fingerprint, idea))
        pool.ideas.append(idea)

    return pool


def enforce_experience_invariant(ideas: Iterable[Idea], context: PromptContext) -> list[Idea]:
    """Drop ideas that need autobiography the knowledge base cannot supply.

    Caught here rather than at draft time on purpose. Once an idea reaches the
    generator the model is being asked to write something it has no material
    for, and the most likely way it obliges is by inventing the material.
    Removing the request is safer than rejecting the output.
    """
    verified = [e for e in context.experiences if e.is_usable_for_first_person]
    result: list[Idea] = []
    for idea in ideas:
        if idea.needs_experience and not verified:
            idea.rejected_because = "needs an experience that is not verified"
        elif idea.needs_experience:
            # Attach the first verified experience so the draft carries the
            # link the sanitiser checks against.
            idea.experience_id = verified[0].id
        result.append(idea)
    return result


def rank(ideas: Iterable[Idea]) -> list[Idea]:
    """Strongest first.

    Novelty leads, then technical depth, because a shallow novel idea makes a
    thinner post than a deep one. Ties break on topic so the ordering is
    stable across runs and a rerun does not reshuffle the week.
    """
    depth_rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        ideas,
        key=lambda idea: (-idea.novelty, depth_rank.get(idea.technical_depth, 1), idea.topic),
    )


def parse_ideas(payload: dict, platform: Platform) -> list[Idea]:
    """Turn a structured model response into ideas.

    Tolerant of missing optional fields and skips malformed entries rather
    than failing the batch: losing one idea out of forty is cheap, losing the
    run is not.
    """
    parsed: list[Idea] = []
    for raw in payload.get("ideas", []):
        if not isinstance(raw, dict):
            continue
        topic = str(raw.get("topic", "")).strip()
        angle = str(raw.get("angle", "")).strip()
        if not topic or not angle:
            continue
        try:
            novelty = float(raw.get("novelty", 5.0))
        except (TypeError, ValueError):
            novelty = 5.0
        parsed.append(
            Idea(
                topic=topic,
                angle=angle,
                why_interesting=str(raw.get("why_interesting", "")).strip(),
                content_type=str(raw.get("content_type", "technical")).strip(),
                platform=platform,
                technical_depth=str(raw.get("technical_depth", "medium")).strip(),
                novelty=max(0.0, min(10.0, novelty)),
                needs_experience=bool(raw.get("needs_experience", False)),
            )
        )
    return parsed


def generate_ideas(
    provider: LLMProvider,
    context: PromptContext,
    *,
    content_types: dict[str, int],
    oversample: float = 2.0,
    model: str | None = None,
    effort: str | None = None,
) -> IdeaPool:
    """Generate, filter and rank a pool of ideas.

    ``oversample`` asks for more than needed so the weakest can be dropped.
    The pool is then deduplicated by angle and stripped of anything requiring
    autobiography that does not exist.
    """
    wanted = sum(content_types.values())
    ask = max(wanted, int(wanted * oversample))

    request = idea_request(
        context,
        count=ask,
        content_types=content_types,
        model=model,
        effort=effort,
    )
    payload = provider.complete_json(request)

    ideas = parse_ideas(payload, context.platform)
    ideas = enforce_experience_invariant(ideas, context)

    unusable = [idea for idea in ideas if not idea.usable]
    candidates = rank([idea for idea in ideas if idea.usable])

    pool = deduplicate(candidates)
    pool.dropped.extend(unusable)
    return pool


def select(pool: IdeaPool, content_types: dict[str, int]) -> list[Idea]:
    """Pick the strongest ideas, respecting the target content mix.

    Falls back to filling any shortfall from whatever is left rather than
    returning fewer posts than asked for. A week that is slightly off the
    target mix beats a week with gaps in it.
    """
    remaining = dict(content_types)
    chosen: list[Idea] = []
    leftover: list[Idea] = []

    for idea in pool.usable:
        wanted = remaining.get(idea.content_type, 0)
        if wanted > 0:
            remaining[idea.content_type] = wanted - 1
            chosen.append(idea)
        else:
            leftover.append(idea)

    shortfall = sum(content_types.values()) - len(chosen)
    if shortfall > 0:
        chosen.extend(leftover[:shortfall])
    return chosen
