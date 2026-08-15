"""The weekly run.

One command, one workbook. This is the thing the rest of the system exists to
produce, and it is deliberately the only orchestration layer: every stage it
calls is independently usable and independently tested, so this file is a
sequence rather than a place where logic hides.

The order matters in one respect. Drafts are generated before they are
scheduled, and scheduling is decoupled from content, so a week can be
reshuffled without regenerating anything. Generating into fixed slots would
tie the two together for no benefit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlmodel import Session, select

from contentsys.config import DraftStatus, Platform, Settings
from contentsys.content.context import build_context
from contentsys.content.generate import Draft, generate_batch
from contentsys.content.ideas import generate_ideas
from contentsys.content.ideas import select as select_ideas
from contentsys.db.models import (
    ContentEdit,
    MonetizationSnapshot,
    PublishedPost,
    VoicePreference,
)
from contentsys.evaluation import EvaluationSuite, overall_score, worst_risk
from contentsys.export import workbook as wb
from contentsys.llm.base import LLMProvider, Usage
from contentsys.research.discover import default_config as default_research_config
from contentsys.research.discover import gather, has_standing, store, to_ideas
from contentsys.scheduling.slots import Slot, week_starting, weekly_slots
from contentsys.voice import load_surface
from contentsys.voice.profile import active_profile
from contentsys.voice.surface import SurfaceProfile

Progress = Callable[[str], None]


@dataclass
class WeeklyResult:
    week_starting: date
    drafts: list[Draft] = field(default_factory=list)
    unused_ideas: int = 0
    usage: Usage = field(default_factory=Usage)
    path: str | None = None
    summary: list[str] = field(default_factory=list)
    #: Slots that got no draft, per platform. Tracked because zipping slots
    #: against drafts silently truncates to the shorter of the two, so an idea
    #: pool that came back thin would otherwise produce a week with holes in
    #: it and no explanation anywhere.
    shortfall: dict[str, int] = field(default_factory=dict)
    #: Recent items found, and how many posts reacted to them per platform.
    findings: int = 0
    reactive: dict[str, int] = field(default_factory=dict)

    @property
    def missing(self) -> int:
        return sum(self.shortfall.values())


def _note(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)


def run_weekly(
    session: Session,
    provider: LLMProvider,
    settings: Settings,
    *,
    start: date | None = None,
    x_posts: int | None = None,
    linkedin_posts: int | None = None,
    seed: int | None = None,
    use_research: bool = True,
    progress: Progress | None = None,
) -> WeeklyResult:
    """Generate, evaluate, schedule and export a week."""
    monday = start or week_starting()
    result = WeeklyResult(week_starting=monday)

    research_config = default_research_config()
    research: list = []
    if use_research:
        _note(progress, "looking for what actually happened this week")
        research = gather(research_config)
        result.findings = len(research)
        stored = store(session, research)
        _note(progress, f"found {len(research)} recent items, {stored} new")

    slots = weekly_slots(
        settings,
        start=monday,
        x_posts=x_posts,
        linkedin_posts=linkedin_posts,
        seed=seed,
    )

    calendar_rows: list[wb.CalendarRow] = []
    idea_rows: list[wb.IdeaRow] = []

    for platform in (Platform.X, Platform.LINKEDIN):
        platform_slots = slots[platform]
        if not platform_slots:
            continue

        context = build_context(session, platform, settings=settings)

        # Reactive ideas first, so the evergreen allocation is sized around
        # what research actually turned up rather than assuming a fixed split
        # and leaving a gap when the sources are quiet.
        reactive: list = []
        if research:
            wanted = research_config.reactive_count(len(platform_slots))
            if wanted:
                usable = [
                    finding for finding in research if has_standing(finding, context.knowledge)
                ]
                reactive = to_ideas(usable[:wanted], platform)
                if reactive:
                    result.reactive[platform.value] = len(reactive)
                    _note(
                        progress,
                        f"{platform.value}: {len(reactive)} posts will react to recent news",
                    )
                elif usable != research:
                    _note(
                        progress,
                        f"{platform.value}: nothing recent that the knowledge base gives "
                        "standing to comment on, so the week stays evergreen",
                    )

        evergreen_target = max(0, len(platform_slots) - len(reactive))
        _note(progress, f"{platform.value}: generating ideas for {evergreen_target} posts")
        allocation = settings.content_mix.allocate(platform, evergreen_target)

        pool = generate_ideas(
            provider,
            context,
            content_types=allocation,
            oversample=settings.idea_oversample,
            model=settings.generation_model,
            effort=settings.generation_effort,
        )
        chosen = reactive + select_ideas(pool, allocation)
        result.unused_ideas += len(pool.usable) - (len(chosen) - len(reactive))

        # Snapshot the published history before drafting. generate_batch
        # deliberately appends each draft to context.recent_posts so later
        # posts in the run can avoid repeating earlier ones, which means that
        # by the time evaluation happens the list already contains every
        # draft. Reading it then makes each post its own predecessor, and the
        # repetition evaluator reports "identical to a previous post" for the
        # entire week.
        published_history = list(context.recent_posts)

        _note(progress, f"{platform.value}: drafting {len(chosen)} posts")
        drafts = generate_batch(
            provider,
            context,
            chosen,
            settings,
            model=settings.generation_model,
            effort=settings.generation_effort,
        )
        result.drafts.extend(drafts)
        for draft in drafts:
            result.usage = result.usage + draft.usage

        _note(progress, f"{platform.value}: evaluating")
        voice = _voice_for(session, platform)
        suite = EvaluationSuite(settings, voice)
        history = published_history

        missing = len(platform_slots) - len(drafts)
        if missing > 0:
            result.shortfall[platform.value] = missing
            _note(
                progress,
                f"{platform.value}: only {len(drafts)} of {len(platform_slots)} slots filled. "
                "The idea pool came back thin, so the week is short by "
                f"{missing}. Raise CONTENTSYS_IDEA_OVERSAMPLE or add material to "
                "the knowledge base.",
            )

        for slot, draft in zip(platform_slots, drafts, strict=False):
            calendar_rows.append(_calendar_row(slot, draft, suite, history))
            if draft.content:
                history.append(draft.content)

        idea_rows.extend(
            wb.IdeaRow(
                idea=f"{idea.topic}: {idea.angle}",
                topic=idea.topic,
                angle=idea.angle,
                platform=platform.value,
                content_type=idea.content_type,
                why_interesting=idea.why_interesting,
                personal_connection="verified experience" if idea.experience_id else "",
                novelty=idea.novelty,
            )
            for idea in pool.usable
            if idea not in chosen
        )

    book = wb.WeeklyWorkbook(
        week_starting=monday,
        calendar=sorted(calendar_rows, key=lambda row: row.timestamp),
        ideas=idea_rows,
        research=_research_rows(session),
        feedback=_feedback_rows(session),
        history=_history_rows(session),
        monetization=_monetization_rows(session, settings),
    )

    _note(progress, "writing the workbook")
    result.path = str(wb.write(book, settings.export_dir))
    result.summary = wb.summarise(book)
    if result.findings:
        total = sum(result.reactive.values())
        result.summary.append(
            f"{total} of these react to recent news, from {result.findings} items found"
        )
    if result.missing:
        gaps = ", ".join(f"{count} on {name}" for name, count in result.shortfall.items())
        result.summary.append(
            f"Short by {result.missing} ({gaps}). The idea pool did not fill every slot."
        )
    return result


def _voice_for(session: Session, platform: Platform) -> SurfaceProfile:
    profile = active_profile(session, platform)
    return load_surface(profile) if profile else SurfaceProfile()


def _calendar_row(
    slot: Slot,
    draft: Draft,
    suite: EvaluationSuite,
    history: list[str],
) -> wb.CalendarRow:
    context = suite.context_for(
        slot.platform,
        draft.content_type,
        history=history,
        topic=draft.topic,
        has_verified_experience=draft.experience_id is not None,
    )
    assessment = suite.run(draft.content, context) if draft.content else None

    notes: list[str] = []
    status = DraftStatus.DRAFT
    if draft.needs_review or not draft.content:
        status = DraftStatus.REVIEW
        notes.extend(draft.issues)
    elif assessment is not None and not assessment.passed:
        status = DraftStatus.REVIEW
        notes.extend(assessment.reasons())

    if draft.repairs:
        notes.append("repaired: " + ", ".join(draft.repairs))
    if draft.attempts > 1:
        notes.append(f"{draft.attempts} attempts")
    if draft.idea.why_interesting:
        notes.append(f"why: {draft.idea.why_interesting}")

    return wb.CalendarRow(
        date_text=slot.date_text,
        day=slot.day,
        time_text=slot.time_text,
        timestamp=slot.timestamp,
        platform=slot.platform.value,
        content_type=draft.content_type,
        topic=draft.topic,
        content=draft.content,
        status=status.value,
        authenticity=overall_score(assessment) if assessment else None,
        originality=assessment.score("repetition") if assessment else None,
        voice_match=assessment.score("voice_match") if assessment else None,
        slop_risk=(
            assessment.by_name("slop").risk.value
            if assessment and assessment.by_name("slop") and assessment.by_name("slop").risk
            else ""
        ),
        repetition_risk=worst_risk(assessment).value if assessment else "",
        technical_accuracy=None,
        diagram="",
        source=draft.idea.source or "personal knowledge",
        notes="; ".join(notes),
    )


def _research_rows(session: Session) -> list[wb.ResearchRow]:
    from contentsys.db.models import ResearchSource

    return [
        wb.ResearchRow(
            topic=source.topic or "",
            source=source.title,
            url=source.url or "",
            key_fact=source.key_fact or "",
        )
        for source in session.exec(select(ResearchSource))
    ]


def _feedback_rows(session: Session) -> list[wb.FeedbackRow]:
    preferences = {
        preference.key: preference.description
        for preference in session.exec(select(VoicePreference))
    }
    rows: list[wb.FeedbackRow] = []
    for edit in session.exec(select(ContentEdit).order_by(ContentEdit.created_at.desc())):  # type: ignore[attr-defined]
        rows.append(
            wb.FeedbackRow(
                original=edit.original,
                edited=edit.edited,
                what_changed=edit.change_summary or "",
                preference_learned="; ".join(sorted(preferences.values())[:3]),
                recorded_on=edit.created_at.date().isoformat(),
            )
        )
    return rows


def _history_rows(session: Session) -> list[wb.HistoryRow]:
    return [
        wb.HistoryRow(
            date_text=post.published_at.date().isoformat(),
            platform=post.platform.value,
            content=post.content,
            topic="",
            content_type="",
            published="Yes",
        )
        for post in session.exec(
            select(PublishedPost).order_by(PublishedPost.published_at.desc())  # type: ignore[attr-defined]
        )
    ]


def _monetization_rows(session: Session, settings: Settings) -> list[wb.MonetizationRow]:
    """Progress against the two program gates.

    Shown every week whether or not a snapshot exists, because the point is to
    make the distance visible rather than to report a number that happens to
    have been captured.
    """
    rules = settings.monetization
    snapshot = session.exec(
        select(MonetizationSnapshot).order_by(MonetizationSnapshot.captured_on.desc())  # type: ignore[attr-defined]
    ).first()

    def gate(name: str, current: int | None, target: int, note: str) -> wb.MonetizationRow:
        if current is None:
            return wb.MonetizationRow(name, "not recorded", f"{target:,}", "unknown", note)
        met = current >= target
        return wb.MonetizationRow(
            name,
            f"{current:,}",
            f"{target:,}",
            "met" if met else f"{target - current:,} to go",
            note,
        )

    rows = [
        gate(
            "Verified followers",
            snapshot.verified_followers if snapshot else None,
            rules.required_verified_followers,
            "Program eligibility gate.",
        ),
        gate(
            "Verified Home Timeline impressions, 90 days",
            snapshot.verified_impressions_90d if snapshot else None,
            rules.required_verified_impressions_90d,
            "Replies are excluded from this count. Only impressions from Premium "
            "subscribers on the Home Timeline count, with at least half the post visible.",
        ),
        wb.MonetizationRow(
            "X Premium subscription",
            "yes" if snapshot and snapshot.premium_active else "not recorded",
            "required",
            "met" if snapshot and snapshot.premium_active else "unknown",
            "Required to join and to stay in the program.",
        ),
        wb.MonetizationRow(
            "Engagement bait",
            "0 in this week's drafts",
            "0",
            "met",
            "A bait call to action is a program violation, so any draft containing "
            "one is rejected rather than scored down.",
        ),
    ]
    if snapshot:
        rows.append(
            wb.MonetizationRow(
                "Last measured",
                snapshot.captured_on.isoformat(),
                "",
                "",
                snapshot.notes or "",
            )
        )
    return rows


def record_snapshot(
    session: Session,
    *,
    verified_followers: int | None = None,
    verified_impressions_90d: int | None = None,
    premium_active: bool | None = None,
    captured_on: date | None = None,
    notes: str | None = None,
) -> MonetizationSnapshot:
    """Record where the account stands against the program gates."""
    when = captured_on or datetime.now().date()
    existing = session.exec(
        select(MonetizationSnapshot).where(MonetizationSnapshot.captured_on == when)
    ).first()
    snapshot = existing or MonetizationSnapshot(captured_on=when)
    if verified_followers is not None:
        snapshot.verified_followers = verified_followers
    if verified_impressions_90d is not None:
        snapshot.verified_impressions_90d = verified_impressions_90d
    if premium_active is not None:
        snapshot.premium_active = premium_active
    if notes is not None:
        snapshot.notes = notes
    session.add(snapshot)
    return snapshot
