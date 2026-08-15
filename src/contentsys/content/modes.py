"""Generation modes.

Eight entry points onto the same engine. They differ in where the idea comes
from, not in how a draft is written or checked, so every mode inherits the
same guarantees.

Brain dump is the one that matters most, and it works differently from the
rest. Every other mode asks the model to produce a thought. Brain dump is
handed a real one and told not to lose it.
"""

from __future__ import annotations

from contentsys.config import Platform, Settings
from contentsys.content.generate import Draft, generate_batch, generate_draft
from contentsys.content.ideas import Idea, IdeaPool, generate_ideas, select
from contentsys.llm.base import LLMProvider, LLMRequest
from contentsys.prompts import PromptContext, build_system

# --------------------------------------------------------------------------
# Idea driven modes
# --------------------------------------------------------------------------


def daily_x(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    posts: int | None = None,
) -> list[Draft]:
    """Mode 1. A day of X posts across the configured content mix."""
    target = posts if posts is not None else (settings.schedule.x.posts_per_day or 10)
    allocation = settings.content_mix.allocate(Platform.X, target)
    pool = generate_ideas(
        provider,
        context,
        content_types=allocation,
        oversample=settings.idea_oversample,
        model=settings.generation_model,
        effort=settings.generation_effort,
    )
    return generate_batch(
        provider,
        context,
        select(pool, allocation),
        settings,
        model=settings.generation_model,
        effort=settings.generation_effort,
    )


def weekly_linkedin(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    posts: int | None = None,
) -> list[Draft]:
    """Mode 2. The week's LinkedIn candidates."""
    target = posts if posts is not None else (settings.schedule.linkedin.posts_per_week or 2)
    allocation = settings.content_mix.allocate(Platform.LINKEDIN, target)
    pool = generate_ideas(
        provider,
        context,
        content_types=allocation,
        oversample=settings.idea_oversample,
        model=settings.generation_model,
        effort=settings.generation_effort,
    )
    return generate_batch(
        provider,
        context,
        select(pool, allocation),
        settings,
        model=settings.generation_model,
        effort=settings.generation_effort,
    )


def from_topic(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    topic: str,
    count: int = 5,
) -> IdeaPool:
    """Mode 3. Ideas for one topic, without drafting them yet.

    Returns ideas rather than drafts on purpose: the value of this mode is
    seeing the angles before committing generation to any of them.
    """
    seeded = PromptContext(
        platform=context.platform,
        voice=context.voice,
        rules=context.rules,
        content_type=context.content_type,
        knowledge=context.knowledge,
        opinions=context.opinions,
        experiences=context.experiences,
        recent_posts=context.recent_posts,
        preferences=[*context.preferences, f"Every idea must be about: {topic}"],
    )
    return generate_ideas(
        provider,
        seeded,
        content_types={context.content_type: count},
        oversample=settings.idea_oversample,
        model=settings.generation_model,
        effort=settings.generation_effort,
    )


# --------------------------------------------------------------------------
# Source driven modes
# --------------------------------------------------------------------------


def _one_off(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    topic: str,
    angle: str,
    content_type: str,
    source: str | None = None,
) -> Draft:
    """Shared path for the modes that already know what the post is about."""
    context.content_type = content_type
    idea = Idea(
        topic=topic,
        angle=angle,
        why_interesting=angle,
        content_type=content_type,
        platform=context.platform,
        source=source,
    )
    return generate_draft(
        provider,
        context,
        idea,
        settings,
        model=settings.generation_model,
        effort=settings.generation_effort,
    )


def from_research(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    title: str,
    summary: str,
    url: str | None = None,
) -> Draft:
    """Mode 4. A post built from a paper or article.

    The source is stated as external fact. It is never allowed to become
    something that happened to the owner, which is the failure this mode is
    most exposed to.
    """
    angle = (
        f"This is external material, not personal experience. Say what it claims, "
        f"then say what you make of it, keeping the two clearly apart.\n\n"
        f"Source: {title}\n{summary}"
    )
    return _one_off(
        provider,
        context,
        settings,
        topic=title,
        angle=angle,
        content_type="research_insight" if context.platform is Platform.LINKEDIN else "research",
        source=url or title,
    )


def reaction(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    event: str,
) -> Draft:
    """Mode 5. A response to something happening.

    The value is the read, not the report, so the angle says so explicitly.
    A reaction post that leads with the news is just news with extra steps.
    """
    angle = (
        "Lead with the interpretation, not the news. Assume the reader already "
        "knows roughly what happened. Only say something you can actually "
        f"support.\n\nWhat happened: {event}"
    )
    return _one_off(
        provider, context, settings, topic=event[:80], angle=angle, content_type="reaction"
    )


def explain(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    concept: str,
) -> Draft:
    """Mode 6. A technical concept made understandable.

    Explicitly not simplification by omission. The instruction is to keep the
    mechanism intact, because a simplification that is no longer true is worse
    than no explanation at all for a reader who goes on to use it.
    """
    angle = (
        f"Explain {concept} to a smart reader who has not studied it. Show the "
        "mechanism, not a summary of the mechanism. Do not remove the part that "
        "makes it true in order to make it shorter. If an analogy is used, say "
        "where the analogy stops holding."
    )
    return _one_off(
        provider,
        context,
        settings,
        topic=concept,
        angle=angle,
        content_type="technical_explanation"
        if context.platform is Platform.LINKEDIN
        else "technical",
    )


def personal(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    experience_id: int | None = None,
) -> Draft:
    """Mode 7. A post from a verified experience.

    Refuses rather than improvises when there is nothing verified to draw on.
    This is the mode where the invariant is most likely to be tested, so it
    fails loudly instead of producing something plausible.
    """
    verified = [e for e in context.experiences if e.is_usable_for_first_person]
    if experience_id is not None:
        verified = [e for e in verified if e.id == experience_id]

    if not verified:
        return Draft(
            idea=Idea(
                topic="personal",
                angle="",
                why_interesting="",
                content_type="personal",
                platform=context.platform,
            ),
            content="",
            platform=context.platform,
            content_type="personal",
            issues=[
                "no verified experience is available for this, so nothing personal "
                "can be written without inventing it"
            ],
            needs_review=True,
        )

    chosen = verified[0]
    context.content_type = "personal"
    idea = Idea(
        topic=chosen.summary[:80],
        angle=(
            "Write from this experience and nothing beyond it. Add no detail that "
            f"is not stated here.\n\n{chosen.summary}\n{chosen.detail or ''}"
        ),
        why_interesting=chosen.summary,
        content_type="personal",
        platform=context.platform,
        needs_experience=True,
        experience_id=chosen.id,
    )
    return generate_draft(
        provider,
        context,
        idea,
        settings,
        model=settings.generation_model,
        effort=settings.generation_effort,
    )


# --------------------------------------------------------------------------
# Brain dump
# --------------------------------------------------------------------------

BRAIN_DUMP_INSTRUCTION = """\
Below is something this person actually thought, written quickly and messily.

Your job is to clean it up just enough to post. Not to improve it. Not to make
it sound more impressive. Not to add a hook, a conclusion, or a lesson.

Keep, without exception:
- the actual claim being made, unchanged
- the order the thought arrives in, if it works
- their words wherever their words are fine
- the register, including lowercase and fragments

Fix only:
- typos and genuinely broken grammar
- a sentence so tangled the meaning is lost
- padding that adds nothing

Do not:
- add an opening line that frames what follows
- add a closing line that summarises it
- replace a concrete detail with an abstraction
- make a tentative thought sound certain
- turn a specific observation into a general principle

If the dump is already postable, return it almost unchanged. That is a
success, not a failure to contribute. The most common way to ruin this is to
do too much.\
"""


def brain_dump(
    provider: LLMProvider,
    context: PromptContext,
    settings: Settings,
    *,
    text: str,
) -> Draft:
    """Mode 8. Messy thoughts into a post, without losing the person.

    This is the mode the whole system is really for. Every other mode asks a
    model to have a thought; this one is handed a real thought and told not to
    lose it. The failure mode is not a bad post, it is a competent post that
    has had the person removed from it, and that failure looks like success
    unless you compare against the input.

    Implemented as a direct request rather than through the idea engine,
    because there is no idea to generate: the idea already exists.
    """
    context.content_type = "random_thought"
    request = LLMRequest(
        system=build_system(context),
        prompt=f"{BRAIN_DUMP_INSTRUCTION}\n\n---\n{text.strip()}\n---\n\nOutput the post only.",
        max_tokens=1024 if context.platform is Platform.X else 2048,
        model=settings.generation_model,
        effort=settings.generation_effort,
        tags=("brain_dump", context.platform.value),
    )

    idea = Idea(
        topic="brain dump",
        angle=text.strip()[:200],
        why_interesting="the owner's own thought",
        content_type="random_thought",
        platform=context.platform,
        source="brain dump",
    )

    from contentsys.content.generate import MAX_LENGTH
    from contentsys.content.sanitize import sanitize

    response = provider.complete(request)
    # A brain dump may legitimately contain a real experience, because the
    # owner wrote it. Their own words about their own life are not an
    # invention, so the experience check does not apply here.
    result = sanitize(
        response.text,
        has_verified_experience=True,
        max_length=MAX_LENGTH.get(context.platform),
    )

    return Draft(
        idea=idea,
        content=result.text,
        platform=context.platform,
        content_type="random_thought",
        repairs=result.repairs,
        issues=list(result.violations),
        usage=response.usage,
    )
