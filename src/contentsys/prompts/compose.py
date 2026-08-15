"""Assembling a prompt from fragments.

The composer owns one thing the fragments deliberately do not know about:
order. Stable sections go first, the cache breakpoint sits at the end of them,
and anything that changes per call goes after.

That ordering is worth being strict about. A weekly run makes roughly a
hundred generation calls whose first several thousand tokens are identical.
Putting a topic or a timestamp in an early fragment silently turns a cached
prefix into a full-price one on every call, and nothing fails, so nobody
notices except the bill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contentsys.config import ContentRules, Platform
from contentsys.db.models import Experience, KnowledgeItem, Opinion
from contentsys.llm.base import LLMRequest, SystemBlock, system_prompt
from contentsys.prompts import fragments
from contentsys.voice.surface import SurfaceProfile


@dataclass
class PromptContext:
    """Everything a generation call needs to know."""

    platform: Platform
    voice: SurfaceProfile
    rules: ContentRules

    content_type: str = "technical"
    knowledge: list[KnowledgeItem] = field(default_factory=list)
    opinions: list[Opinion] = field(default_factory=list)
    experiences: list[Experience] = field(default_factory=list)
    recent_posts: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)

    @property
    def has_verified_experience(self) -> bool:
        return any(e.is_usable_for_first_person for e in self.experiences)


def build_system(context: PromptContext) -> tuple[SystemBlock, ...]:
    """Compose the system prompt, with the cache breakpoint in the right place.

    Blocks before and including the breakpoint must be identical across every
    call in a run. The voice profile qualifies: it is rebuilt between runs, not
    within one. The platform and house rules qualify. Content type does not,
    which is why it sits after.
    """
    stable = [
        fragments.BASE_PERSONA,
        fragments.HOUSE_RULES,
        fragments.SLOP_RULES,
        fragments.voice_fragment(context.voice, context.platform),
        fragments.platform_fragment(context.platform, context.rules),
        fragments.knowledge_fragment(context.knowledge),
        fragments.opinions_fragment(context.opinions),
    ]
    volatile = [
        fragments.preferences_fragment(context.preferences),
        fragments.content_type_fragment(context.content_type),
        fragments.experiences_fragment(context.experiences),
        fragments.avoid_repetition_fragment(context.recent_posts),
    ]

    present_stable = [section for section in stable if section]
    present_volatile = [section for section in volatile if section]

    # The breakpoint is the last stable block. Computed rather than hardcoded,
    # because an empty knowledge or opinions section shifts the index.
    return system_prompt(
        *present_stable,
        *present_volatile,
        cache_through=len(present_stable) - 1 if present_stable else None,
    )


def draft_request(
    context: PromptContext,
    *,
    topic: str,
    angle: str,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int = 1024,
    feedback: str | None = None,
) -> LLMRequest:
    """A request for one draft.

    ``feedback`` carries the reason a previous attempt failed. Feeding the
    specific failure back is what makes regeneration converge; regenerating
    with the identical prompt just rolls the dice again.
    """
    parts = [f"Topic: {topic}", f"Angle: {angle}"]
    if feedback:
        parts.append(
            "A previous attempt at this was rejected. Fix exactly this and change "
            f"nothing else about the approach:\n{feedback}"
        )
    parts.append("Write the post now. Output only the post text.")

    return LLMRequest(
        system=build_system(context),
        prompt="\n\n".join(parts),
        max_tokens=max_tokens,
        model=model,
        effort=effort,
        tags=("draft", context.platform.value, context.content_type),
    )


#: Shape of one generated idea. Kept next to the composer because the prompt
#: and the schema have to agree, and separating them invites drift.
IDEA_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "minItems": 1,
            "maxItems": 40,
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "angle": {"type": "string"},
                    "why_interesting": {"type": "string"},
                    "content_type": {"type": "string"},
                    "technical_depth": {"type": "string", "enum": ["low", "medium", "high"]},
                    "novelty": {"type": "number", "minimum": 0, "maximum": 10},
                    "needs_experience": {"type": "boolean"},
                },
                "required": [
                    "topic",
                    "angle",
                    "why_interesting",
                    "content_type",
                    "technical_depth",
                    "novelty",
                    "needs_experience",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ideas"],
    "additionalProperties": False,
}


def idea_request(
    context: PromptContext,
    *,
    count: int,
    content_types: dict[str, int],
    model: str | None = None,
    effort: str | None = None,
) -> LLMRequest:
    """A request for a pool of ideas.

    Ideas are generated before drafts and deliberately oversampled, so the
    weak ones can be dropped rather than written up. Writing 70 posts from 70
    ideas means writing up 70 mediocre ideas.
    """
    wanted = "\n".join(f"- {name}: {n}" for name, n in sorted(content_types.items()) if n)
    prompt = f"""\
Generate {count} distinct post ideas for this person.

Spread them across these content types:
{wanted}

An idea is not a post. It is a topic, plus the specific angle that makes it
worth writing, plus why it is interesting.

What makes an idea good here:
- It comes from something this person actually knows or thinks, listed above.
- It has a specific angle, not a subject area. "Sumcheck" is not an idea.
  "Sumcheck is the only part of the stack that got simpler as it got more
  general" is an idea.
- It could not be written by someone else with the same reading list.

Set needs_experience to true only if the idea requires a first person claim
about having done something. If the verified experience list above cannot
support it, do not generate that idea at all.

Do not repeat angles between ideas. Two ideas on the same topic must differ in
what they claim, not just in wording.\
"""
    return LLMRequest(
        system=build_system(context),
        prompt=prompt,
        max_tokens=8192,
        model=model,
        effort=effort,
        json_schema=IDEA_SCHEMA,
        tags=("ideas", context.platform.value),
    )
