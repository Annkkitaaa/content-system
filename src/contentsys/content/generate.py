"""The draft engine.

One idea in, one checked draft out, with a bounded regeneration loop in
between.

The loop is the part worth explaining. Regenerating with the identical prompt
is just rolling the dice again, so every retry carries the specific reason the
previous attempt failed. And the loop is bounded: after the configured number
of attempts a draft is handed back flagged for review rather than dropped or
silently shipped. Both of those alternatives are worse. Dropping it leaves a
hole in the week with no explanation, and shipping it quietly is how a system
that promises authenticity starts publishing things nobody checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contentsys.config import Platform, Settings
from contentsys.content.ideas import Idea
from contentsys.content.sanitize import SanitizeResult, sanitize
from contentsys.llm.base import LLMError, LLMProvider, LLMRefusal, Usage
from contentsys.prompts import PromptContext, draft_request
from contentsys.voice.surface import compare

#: Platform ceilings. X is a hard API limit; the LinkedIn number is a
#: readability judgement, not a platform constraint.
MAX_LENGTH: dict[Platform, int] = {
    Platform.X: 280,
    Platform.LINKEDIN: 3000,
}


@dataclass
class Draft:
    """A generated post and everything known about it."""

    idea: Idea
    content: str
    platform: Platform
    content_type: str
    attempts: int = 1
    experience_id: int | None = None
    repairs: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    #: True when the loop ran out of attempts. The draft is returned anyway,
    #: because a flagged draft the owner can fix beats a silent gap.
    needs_review: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def topic(self) -> str:
        return self.idea.topic


def _voice_feedback(context: PromptContext, text: str) -> list[str]:
    """Mechanical voice deviations, as instructions rather than a score.

    A score without a reason cannot be fed back into a retry, so this returns
    the specific things that are off.
    """
    return compare(context.voice, text)["issues"]


def generate_draft(
    provider: LLMProvider,
    context: PromptContext,
    idea: Idea,
    settings: Settings,
    *,
    model: str | None = None,
    effort: str | None = None,
) -> Draft:
    """Generate one draft, retrying with specific feedback until it passes."""
    max_length = MAX_LENGTH.get(context.platform)
    attempts = settings.thresholds.max_regeneration_attempts + 1
    feedback: str | None = None
    usage = Usage()
    last: Draft | None = None

    for attempt in range(1, attempts + 1):
        request = draft_request(
            context,
            topic=idea.topic,
            angle=idea.angle,
            model=model,
            effort=effort,
            max_tokens=1024 if context.platform is Platform.X else 2048,
            feedback=feedback,
        )

        try:
            response = provider.complete(request)
        except LLMRefusal as exc:
            # A refusal fails identically on retry, so looping burns quota for
            # nothing. Hand it back flagged instead.
            return Draft(
                idea=idea,
                content="",
                platform=context.platform,
                content_type=idea.content_type,
                attempts=attempt,
                issues=[f"the model declined this topic: {exc}"],
                needs_review=True,
                usage=usage,
            )
        except LLMError as exc:
            if attempt >= attempts:
                return Draft(
                    idea=idea,
                    content="",
                    platform=context.platform,
                    content_type=idea.content_type,
                    attempts=attempt,
                    issues=[f"generation failed: {exc}"],
                    needs_review=True,
                    usage=usage,
                )
            feedback = None
            continue

        usage = usage + response.usage

        result: SanitizeResult = sanitize(
            response.text,
            has_verified_experience=idea.experience_id is not None,
            max_length=max_length,
        )
        issues = list(result.violations) + _voice_feedback(context, result.text)

        last = Draft(
            idea=idea,
            content=result.text,
            platform=context.platform,
            content_type=idea.content_type,
            attempts=attempt,
            experience_id=idea.experience_id,
            repairs=result.repairs,
            issues=issues,
            usage=usage,
        )
        if not issues:
            return last

        feedback = "\n".join(f"- {issue}" for issue in issues)

    assert last is not None
    last.needs_review = True
    return last


def generate_batch(
    provider: LLMProvider,
    context: PromptContext,
    ideas: list[Idea],
    settings: Settings,
    *,
    model: str | None = None,
    effort: str | None = None,
    on_progress: object = None,
) -> list[Draft]:
    """Generate a draft per idea.

    Each draft is added to the context's recent list as it is produced, so
    later drafts in the same batch can see and avoid what earlier ones said.
    Without that, a single run happily writes the same post five times.
    """
    drafts: list[Draft] = []
    for index, idea in enumerate(ideas, start=1):
        context.content_type = idea.content_type
        draft = generate_draft(provider, context, idea, settings, model=model, effort=effort)
        drafts.append(draft)
        if draft.content:
            context.recent_posts.append(draft.content)
        if callable(on_progress):
            on_progress(index, len(ideas), draft)
    return drafts
