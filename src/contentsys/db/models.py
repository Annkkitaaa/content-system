"""Database schema.

Three ideas shape this schema.

**The identity layer is separate from everything else.** ``Experience``,
``Opinion`` and ``KnowledgeItem`` describe the owner. ``ResearchSource``
describes the world. ``ContentDraft`` is generated interpretation built from
both. A draft that claims a personal experience must carry a real
``experience_id``, which is why that link is a foreign key rather than a
sentence in a prompt.

**Evaluations are rows, not columns.** One row per draft per evaluator, so a
score is auditable, comparable over time, and extensible without a migration
every time an evaluator is added.

**Scheduling is decoupled from content.** A ``ScheduleSlot`` points at a
draft rather than a draft owning a timestamp, so the week can be reshuffled
without regenerating anything.

Note on style: this is the one module without ``from __future__ import
annotations``. SQLModel resolves ``Relationship`` targets through SQLAlchemy's
class registry rather than through ``typing``, and PEP 563 turns
``list["ContentDraft"]`` into a string the registry cannot parse. Python 3.12
gives us ``int | None`` natively anyway, so nothing is lost.
"""

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel

from contentsys.config import DraftStatus, Platform


def _now() -> datetime:
    return datetime.now(UTC)


class SampleSource(StrEnum):
    """Where a writing sample came from.

    Worth distinguishing because register differs sharply by platform. The
    same person writes in lowercase fragments on X and in structured
    paragraphs on Medium, and collapsing those into one voice model produces
    something that sounds like neither.
    """

    X = "x"
    LINKEDIN = "linkedin"
    MEDIUM = "medium"
    NOTE = "note"
    OTHER = "other"


class Confidence(StrEnum):
    """How sure we are that something is true about the owner.

    ``STATED`` means the owner said it directly. ``INFERRED`` means the
    system derived it and it has not been confirmed. Only ``STATED`` facts
    may back a first-person claim in content.
    """

    STATED = "stated"
    INFERRED = "inferred"


class TimestampMixin(SQLModel):
    created_at: datetime = Field(default_factory=_now, nullable=False)
    updated_at: datetime = Field(default_factory=_now, nullable=False)


# --------------------------------------------------------------------------
# Identity layer: who the owner is
# --------------------------------------------------------------------------


class WritingSample(TimestampMixin, table=True):
    """A piece the owner actually wrote.

    This is the ground truth for the voice engine. Nothing generated ever
    goes in here, because a voice model trained on its own output drifts
    toward the average of itself.
    """

    __tablename__ = "writing_samples"

    id: int | None = Field(default=None, primary_key=True)
    source: SampleSource = Field(index=True)
    content: str
    #: Stable hash of the normalised text, so re-importing an export does not
    #: duplicate every sample.
    fingerprint: str = Field(index=True, unique=True)
    published_at: date | None = Field(default=None, index=True)
    url: str | None = None
    topic: str | None = Field(default=None, index=True)

    #: Real engagement, where known. Used later to look for patterns, never
    #: to decide what is worth writing.
    impressions: int | None = None
    likes: int | None = None
    replies: int | None = None

    #: Excluded from voice analysis. Set for things that are the owner's
    #: words but not representative writing, such as a one word reply.
    excluded: bool = Field(default=False)
    exclusion_reason: str | None = None


class Experience(TimestampMixin, table=True):
    """Something that actually happened to the owner.

    The single most important table in the system. Content may only make a
    first-person experience claim when it links to a row here. If it is not
    in this table, it did not happen, and the generator writes something
    analytical instead of inventing a story.
    """

    __tablename__ = "experiences"

    id: int | None = Field(default=None, primary_key=True)
    summary: str
    detail: str | None = None
    organisation: str | None = Field(default=None, index=True)
    role: str | None = None
    started_on: date | None = None
    ended_on: date | None = None
    #: Verbatim proof, so a claim can always be traced back to its source.
    evidence: str | None = None
    evidence_url: str | None = None
    confidence: Confidence = Field(default=Confidence.STATED, index=True)
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Quoted deliberately. See the UP037 note in pyproject.toml.
    drafts: list["ContentDraft"] = Relationship(back_populates="experience")

    @property
    def is_usable_for_first_person(self) -> bool:
        return self.confidence is Confidence.STATED


class Opinion(TimestampMixin, table=True):
    """Something the owner actually thinks.

    ``strength`` matters: content should not present a tentative view with
    the confidence of a settled one. ``uncertain`` opinions are usable, but
    only phrased as open questions.
    """

    __tablename__ = "opinions"

    id: int | None = Field(default=None, primary_key=True)
    statement: str
    reasoning: str | None = None
    topic: str | None = Field(default=None, index=True)
    #: strong, held, tentative, uncertain
    strength: str = Field(default="held", index=True)
    evidence: str | None = None
    confidence: Confidence = Field(default=Confidence.STATED)


class KnowledgeItem(TimestampMixin, table=True):
    """Something the owner demonstrably understands.

    Depth is what stops the system writing a confident post about a topic the
    owner has only skimmed. A concept marked ``familiar`` gets hedged
    language; one marked ``deep`` can carry a strong claim.
    """

    __tablename__ = "knowledge_items"

    id: int | None = Field(default=None, primary_key=True)
    concept: str = Field(index=True)
    domain: str = Field(index=True)
    #: aware, familiar, working, deep
    depth: str = Field(default="familiar", index=True)
    notes: str | None = None
    evidence: str | None = None
    evidence_url: str | None = None
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class Topic(TimestampMixin, table=True):
    """A subject area the owner writes about.

    ``is_core`` drives niche coherence. Ranking on X is interest-graph
    driven, so a week that wanders too far outside the core set costs
    distribution.
    """

    __tablename__ = "topics"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    is_core: bool = Field(default=True, index=True)
    #: Days before this topic may be covered again.
    cooldown_days: int = Field(default=5)
    avoid: bool = Field(default=False)
    avoid_reason: str | None = None


class VoiceProfile(TimestampMixin, table=True):
    """A snapshot of how the owner writes.

    Versioned rather than mutated, so a regression in generated voice can be
    traced to the profile that produced it and rolled back.

    ``surface`` holds deterministic measurements (sentence length, lowercase
    ratio, punctuation habits). ``semantic`` holds the extracted model of how
    the owner thinks and argues. The second one drives generation; the first
    is a post-generation check.
    """

    __tablename__ = "voice_profiles"
    __table_args__ = (UniqueConstraint("platform", "version"),)

    id: int | None = Field(default=None, primary_key=True)
    platform: Platform = Field(index=True)
    version: int = Field(default=1)
    is_active: bool = Field(default=True, index=True)
    sample_count: int = Field(default=0)
    surface: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    semantic: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class VoicePreference(TimestampMixin, table=True):
    """Something learned from how the owner edits drafts.

    ``confidence`` increments each time the same pattern is observed. Only
    preferences above a floor reach the prompt, so one unusual edit does not
    permanently reshape the voice model.
    """

    __tablename__ = "voice_preferences"

    id: int | None = Field(default=None, primary_key=True)
    #: A short stable key, such as "prefers_lowercase_openers".
    key: str = Field(index=True, unique=True)
    description: str
    #: Positive means do more of this, negative means do less.
    polarity: int = Field(default=1)
    confidence: int = Field(default=1)
    platform: Platform | None = Field(default=None, index=True)
    examples: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    active: bool = Field(default=True, index=True)


# --------------------------------------------------------------------------
# External layer: what happened in the world
# --------------------------------------------------------------------------


class ResearchSource(TimestampMixin, table=True):
    """An external fact, with a citation.

    Never a personal experience. A fact from here can inform a draft, but the
    draft says "this happened", not "this happened to me".
    """

    __tablename__ = "research_sources"

    id: int | None = Field(default=None, primary_key=True)
    title: str
    url: str | None = Field(default=None, index=True)
    kind: str = Field(default="article", index=True)
    key_fact: str | None = None
    fetched_at: datetime = Field(default_factory=_now)
    topic: str | None = Field(default=None, index=True)


# --------------------------------------------------------------------------
# Generated layer
# --------------------------------------------------------------------------


class GenerationRun(TimestampMixin, table=True):
    """One weekly batch.

    Groups everything a single run produced so a bad week can be inspected,
    compared or discarded as a unit.
    """

    __tablename__ = "generation_runs"

    id: int | None = Field(default=None, primary_key=True)
    week_starting: date = Field(index=True)
    provider: str
    generation_model: str
    notes: str | None = None
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    cache_read_tokens: int = Field(default=0)

    drafts: list["ContentDraft"] = Relationship(back_populates="run")


class Idea(TimestampMixin, table=True):
    """A candidate before it becomes a post.

    Ideas are generated separately and oversampled, so the weak ones can be
    dropped rather than written up. Writing 70 posts from 70 ideas means
    writing up 70 mediocre ideas.
    """

    __tablename__ = "ideas"

    id: int | None = Field(default=None, primary_key=True)
    run_id: int | None = Field(default=None, foreign_key="generation_runs.id", index=True)
    topic: str = Field(index=True)
    angle: str
    why_interesting: str
    content_type: str = Field(index=True)
    platform: Platform = Field(index=True)
    source: str | None = None
    #: Set only when the idea draws on a verified experience.
    experience_id: int | None = Field(default=None, foreign_key="experiences.id")
    technical_depth: str = Field(default="medium")
    novelty: float = Field(default=5.0)
    used: bool = Field(default=False, index=True)


class ContentDraft(TimestampMixin, table=True):
    """A generated post."""

    __tablename__ = "content_drafts"

    id: int | None = Field(default=None, primary_key=True)
    run_id: int | None = Field(default=None, foreign_key="generation_runs.id", index=True)
    idea_id: int | None = Field(default=None, foreign_key="ideas.id", index=True)

    platform: Platform = Field(index=True)
    content_type: str = Field(index=True)
    topic: str = Field(index=True)
    content: str
    status: DraftStatus = Field(default=DraftStatus.DRAFT, index=True)

    #: The invariant. A draft making a first-person experience claim must
    #: point at the experience that backs it.
    experience_id: int | None = Field(default=None, foreign_key="experiences.id", index=True)
    #: primary_work or commentary. Recorded per post because the X program
    #: pays on original content and treats the distinction as material.
    provenance: str = Field(default="primary_work", index=True)
    source_note: str | None = None

    #: Fingerprint of the post's structure, for structural repetition checks.
    structure_fingerprint: str | None = Field(default=None, index=True)
    generation_attempt: int = Field(default=1)

    run: GenerationRun | None = Relationship(back_populates="drafts")
    experience: Experience | None = Relationship(back_populates="drafts")
    evaluations: list["Evaluation"] = Relationship(back_populates="draft")
    visuals: list["Visual"] = Relationship(back_populates="draft")


class Evaluation(TimestampMixin, table=True):
    """One evaluator's verdict on one draft."""

    __tablename__ = "evaluations"
    __table_args__ = (UniqueConstraint("draft_id", "evaluator", "attempt"),)

    id: int | None = Field(default=None, primary_key=True)
    draft_id: int = Field(foreign_key="content_drafts.id", index=True)
    evaluator: str = Field(index=True)
    attempt: int = Field(default=1)
    score: float | None = None
    verdict: str | None = None
    #: True when this evaluator alone is enough to reject the draft, which is
    #: how an engagement bait finding behaves: it is a program violation, not
    #: a score to average away.
    blocking: bool = Field(default=False)
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    draft: ContentDraft | None = Relationship(back_populates="evaluations")


class Visual(TimestampMixin, table=True):
    """A diagram generated to accompany a post.

    The spec is stored alongside the rendered file so an image can be
    regenerated after a style change without asking a model to reinvent it.
    """

    __tablename__ = "visuals"

    id: int | None = Field(default=None, primary_key=True)
    draft_id: int = Field(foreign_key="content_drafts.id", index=True)
    #: flow, chain, comparison, timeline, plot
    kind: str = Field(default="flow", index=True)
    title: str | None = None
    alt_text: str | None = None
    spec: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    file_path: str | None = None

    draft: ContentDraft | None = Relationship(back_populates="visuals")


class ScheduleSlot(TimestampMixin, table=True):
    """When a draft is meant to go out."""

    __tablename__ = "schedule_slots"

    id: int | None = Field(default=None, primary_key=True)
    draft_id: int = Field(foreign_key="content_drafts.id", index=True, unique=True)
    run_id: int | None = Field(default=None, foreign_key="generation_runs.id", index=True)
    platform: Platform = Field(index=True)
    scheduled_for: datetime = Field(index=True)
    window_name: str | None = None


class ContentEdit(TimestampMixin, table=True):
    """The owner's rewrite of a draft.

    The raw material for voice memory. ``learn`` is opt in, because not every
    edit expresses a preference; some are just fixing a fact.
    """

    __tablename__ = "content_edits"

    id: int | None = Field(default=None, primary_key=True)
    draft_id: int = Field(foreign_key="content_drafts.id", index=True)
    original: str
    edited: str
    learn: bool = Field(default=True, index=True)
    processed: bool = Field(default=False, index=True)
    change_summary: str | None = None


class PublishedPost(TimestampMixin, table=True):
    """A post that actually went out.

    Written only after the owner confirms publication. Nothing in this system
    publishes on its own.
    """

    __tablename__ = "published_posts"

    id: int | None = Field(default=None, primary_key=True)
    draft_id: int | None = Field(default=None, foreign_key="content_drafts.id", index=True)
    platform: Platform = Field(index=True)
    content: str
    published_at: datetime = Field(index=True)
    external_id: str | None = Field(default=None, index=True)
    url: str | None = None


class ContentPerformance(TimestampMixin, table=True):
    """Measured results for a published post.

    ``qualified_impressions`` is tracked separately from raw impressions
    because the X program pays only on unique Home Timeline views from
    Premium subscribers with at least half the post visible. Raw view counts
    correlate poorly with earnings.
    """

    __tablename__ = "content_performance"

    id: int | None = Field(default=None, primary_key=True)
    published_post_id: int = Field(foreign_key="published_posts.id", index=True)
    measured_at: datetime = Field(default_factory=_now, index=True)
    impressions: int | None = None
    qualified_impressions: int | None = None
    likes: int | None = None
    replies: int | None = None
    reposts: int | None = None
    bookmarks: int | None = None
    profile_visits: int | None = None
    follower_delta: int | None = None


class MonetizationSnapshot(TimestampMixin, table=True):
    """Progress toward X Original Content Rewards eligibility.

    Two gates: verified followers and verified Home Timeline impressions over
    a rolling 90 days, with replies excluded from the impression count.
    """

    __tablename__ = "monetization_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    captured_on: date = Field(index=True, unique=True)
    verified_followers: int | None = None
    verified_impressions_90d: int | None = None
    premium_active: bool | None = None
    notes: str | None = None

    def gates(self, *, required_followers: int, required_impressions: int) -> dict[str, bool]:
        return {
            "verified_followers": (self.verified_followers or 0) >= required_followers,
            "verified_impressions_90d": (self.verified_impressions_90d or 0)
            >= required_impressions,
            "premium_active": bool(self.premium_active),
        }
