"""Voice memory.

Turns classified edits into durable preferences that reach future prompts.

The design question here is how fast to learn. Too slow and the system never
adapts. Too fast and one unusual edit permanently distorts how everything is
written, which is worse, because the owner has no easy way to see that it
happened or to undo it.

So: a preference is recorded on the first observation but not used. It reaches
the prompt only once the same change has been seen enough times to be a
pattern rather than a mood. Contradictions decay a preference instead of
fighting it, because an edit going the other way is real evidence too.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlmodel import Session, select

from contentsys.config import Platform
from contentsys.db.models import ContentEdit, VoicePreference
from contentsys.voice.diff import EditAnalysis, analyse_edit

#: Below this a preference is remembered but withheld from the prompt. One
#: edit is a data point; three is a habit.
ACTIVE_CONFIDENCE = 2

#: Confidence cannot climb forever, or an early preference becomes impossible
#: to overturn with later evidence.
MAX_CONFIDENCE = 8

#: Pairs that cannot both be true. Observing one weakens the other rather
#: than leaving the prompt holding both.
_OPPOSITES: dict[str, str] = {
    "prefers_shorter": "prefers_longer",
    "prefers_longer": "prefers_shorter",
    "prefers_lowercase": "prefers_capitalisation",
    "prefers_capitalisation": "prefers_lowercase",
    "adds_emoji": "removes_emoji",
    "removes_emoji": "adds_emoji",
    "softens_claims": "removes_hedging",
    "removes_hedging": "softens_claims",
    "adds_question": "removes_closing_question",
    "removes_closing_question": "adds_question",
}


@dataclass
class LearningReport:
    """What one edit taught."""

    analysis: EditAnalysis
    learned: list[str]
    reinforced: list[str]
    weakened: list[str]
    now_active: list[str]

    def describe(self) -> str:
        if not self.analysis.substantive:
            return "nothing substantive changed, so nothing was learned"
        parts = []
        if self.learned:
            parts.append(f"new: {', '.join(self.learned)}")
        if self.reinforced:
            parts.append(f"reinforced: {', '.join(self.reinforced)}")
        if self.weakened:
            parts.append(f"weakened: {', '.join(self.weakened)}")
        if self.now_active:
            parts.append(f"now used in prompts: {', '.join(self.now_active)}")
        return "; ".join(parts) or "observed, nothing changed"


def record_edit(
    session: Session,
    *,
    draft_id: int,
    original: str,
    edited: str,
    learn: bool = True,
) -> ContentEdit:
    """Store an edit. Learning from it is a separate, opt in step."""
    edit = ContentEdit(
        draft_id=draft_id,
        original=original,
        edited=edited,
        learn=learn,
        change_summary=analyse_edit(original, edited).summary(),
    )
    session.add(edit)
    return edit


def learn_from(
    session: Session,
    original: str,
    edited: str,
    *,
    platform: Platform | None = None,
) -> LearningReport:
    """Classify an edit and fold it into voice memory."""
    analysis = analyse_edit(original, edited)
    report = LearningReport(
        analysis=analysis, learned=[], reinforced=[], weakened=[], now_active=[]
    )

    if not analysis.substantive:
        return report

    for change in analysis.changes:
        existing = session.exec(
            select(VoicePreference).where(VoicePreference.key == change.key)
        ).first()

        was_active = bool(existing) and existing.confidence >= ACTIVE_CONFIDENCE

        if existing is None:
            preference = VoicePreference(
                key=change.key,
                description=change.description,
                polarity=change.polarity,
                confidence=1,
                platform=platform,
                examples=[change.evidence] if change.evidence else [],
            )
            session.add(preference)
            report.learned.append(change.key)
            existing = preference
        else:
            existing.confidence = min(MAX_CONFIDENCE, existing.confidence + 1)
            existing.active = True
            # Keep the most recent phrasing: descriptions carry a percentage
            # that should reflect the latest evidence rather than the first.
            existing.description = change.description
            if change.evidence and change.evidence not in existing.examples:
                existing.examples = [*existing.examples, change.evidence][-5:]
            session.add(existing)
            report.reinforced.append(change.key)

        if not was_active and existing.confidence >= ACTIVE_CONFIDENCE:
            report.now_active.append(change.key)

        opposite_key = _OPPOSITES.get(change.key)
        if opposite_key:
            opposite = session.exec(
                select(VoicePreference).where(VoicePreference.key == opposite_key)
            ).first()
            if opposite is not None and opposite.confidence > 0:
                # Evidence the other way is still evidence. Decaying rather
                # than deleting means a genuine reversal takes as long to
                # learn as the original did.
                opposite.confidence -= 1
                opposite.active = opposite.confidence > 0
                session.add(opposite)
                report.weakened.append(opposite_key)

    return report


def active_preferences(session: Session, platform: Platform | None = None) -> list[VoicePreference]:
    """Preferences confident enough to reach a prompt."""
    statement = select(VoicePreference).where(
        VoicePreference.active == True,  # noqa: E712 - SQL comparison
        VoicePreference.confidence >= ACTIVE_CONFIDENCE,
    )
    preferences = list(session.exec(statement))
    if platform is not None:
        preferences = [p for p in preferences if p.platform in (None, platform)]
    return sorted(preferences, key=lambda p: (-p.confidence, p.key))


def forget(session: Session, key: str) -> bool:
    """Drop a learned preference.

    Needed because the system will occasionally learn something wrong, and a
    voice model with no undo is one nobody will trust enough to keep feeding.
    """
    preference = session.exec(select(VoicePreference).where(VoicePreference.key == key)).first()
    if preference is None:
        return False
    session.delete(preference)
    return True
