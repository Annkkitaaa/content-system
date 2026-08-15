"""Building a prompt context from the knowledge base.

One place that reads the identity layer and assembles what a generation call
needs. Kept separate from the prompt composer so that composition stays pure
and testable with hand-built context, and separate from the engines so they
never touch the database directly.
"""

from __future__ import annotations

from sqlmodel import Session, select

from contentsys.config import Platform, Settings, get_settings
from contentsys.db.models import (
    Experience,
    KnowledgeItem,
    Opinion,
    PublishedPost,
    VoicePreference,
)
from contentsys.prompts import PromptContext
from contentsys.voice import active_profile, load_surface
from contentsys.voice.surface import SurfaceProfile

#: Preferences below this have been seen too few times to trust. One unusual
#: edit should not permanently reshape how everything is written.
MIN_PREFERENCE_CONFIDENCE = 2

#: How many recent posts to show the model as things not to repeat. Enough to
#: cover a couple of weeks without pushing the prompt past the point where
#: the instruction gets diluted.
RECENT_POST_LIMIT = 30


def build_context(
    session: Session,
    platform: Platform,
    *,
    content_type: str = "technical",
    settings: Settings | None = None,
    topic: str | None = None,
) -> PromptContext:
    """Assemble everything a generation call needs about the owner."""
    settings = settings or get_settings()

    profile = active_profile(session, platform)
    voice: SurfaceProfile = load_surface(profile) if profile else SurfaceProfile()

    knowledge = list(session.exec(select(KnowledgeItem)))
    opinions = list(session.exec(select(Opinion)))
    if topic:
        # Prefer opinions on the topic at hand, but keep some breadth so the
        # model is not boxed into one view.
        relevant = [o for o in opinions if o.topic and o.topic.lower() in topic.lower()]
        others = [o for o in opinions if o not in relevant]
        opinions = relevant + others[: max(0, 8 - len(relevant))]

    experiences = list(session.exec(select(Experience)))

    recent = list(
        session.exec(
            select(PublishedPost)
            .where(PublishedPost.platform == platform)
            .order_by(PublishedPost.published_at.desc())  # type: ignore[attr-defined]
            .limit(RECENT_POST_LIMIT)
        )
    )

    preferences = list(
        session.exec(
            select(VoicePreference).where(
                VoicePreference.active == True,  # noqa: E712 - SQL comparison
                VoicePreference.confidence >= MIN_PREFERENCE_CONFIDENCE,
            )
        )
    )

    return PromptContext(
        platform=platform,
        voice=voice,
        rules=settings.content_rules,
        content_type=content_type,
        knowledge=knowledge,
        opinions=opinions,
        experiences=experiences,
        recent_posts=[post.content for post in recent],
        preferences=[_render_preference(p) for p in preferences],
    )


def _render_preference(preference: VoicePreference) -> str:
    direction = "Do this" if preference.polarity > 0 else "Avoid this"
    seen = f"observed {preference.confidence} times"
    return f"{direction}: {preference.description.strip()} ({seen})"
