"""Running every evaluator and deciding what happens next.

The suite owns two things no individual evaluator knows about: which order to
run them in, and how their verdicts combine into a decision.

Order matters for cost. The deterministic checks are free and catch most
failures, so they run first and short circuit the expensive ones. A draft
carrying engagement bait does not need an authenticity score to know it is
being regenerated.
"""

from __future__ import annotations

from contentsys.config import Platform, Settings, SlopRisk
from contentsys.content.sanitize import find_experience_claims
from contentsys.evaluation.base import Assessment, EvaluationContext, EvaluationResult
from contentsys.evaluation.monetization import MonetizationEvaluator
from contentsys.evaluation.repetition import RepetitionDetector
from contentsys.evaluation.slop import SlopDetector
from contentsys.voice.surface import SurfaceProfile, compare


class VoiceEvaluator:
    """Mechanical voice match against the measured profile."""

    name = "voice_match"

    def __init__(self, profile: SurfaceProfile) -> None:
        self.profile = profile

    def evaluate(self, text: str, context: EvaluationContext) -> EvaluationResult:
        result = compare(self.profile, text)
        issues = result["issues"]
        # Each deviation costs 2.5, so one is a warning and three is a
        # different person.
        score = max(0.0, 10.0 - 2.5 * len(issues))
        return EvaluationResult(
            evaluator=self.name,
            score=score,
            reason="; ".join(issues) if issues else "matches the measured voice",
            details={"issues": issues},
        )


class ExperienceEvaluator:
    """The invariant, as an evaluator rather than only a sanitiser check.

    Duplicated on purpose. The sanitiser catches it during generation; this
    catches it for anything that reaches the suite by another route, such as a
    draft edited by hand and re-scored.
    """

    name = "experience"

    def evaluate(self, text: str, context: EvaluationContext) -> EvaluationResult:
        if context.has_verified_experience:
            return EvaluationResult(
                evaluator=self.name, score=10.0, reason="backed by a verified experience"
            )
        claims = find_experience_claims(text)
        if not claims:
            return EvaluationResult(
                evaluator=self.name, score=10.0, reason="makes no personal claim"
            )
        return EvaluationResult(
            evaluator=self.name,
            score=0.0,
            blocking=True,
            reason=(
                "claims a personal experience with nothing in the knowledge base behind it: "
                + ", ".join(sorted(set(claims))[:3])
            ),
            details={"claims": claims},
        )


class EvaluationSuite:
    """Every check, in cost order, with the pass or fail decision."""

    def __init__(self, settings: Settings, voice: SurfaceProfile) -> None:
        self.settings = settings
        self.thresholds = settings.thresholds
        self.evaluators = [
            # Free and conclusive first. A blocking failure here means the
            # expensive checks never run.
            ExperienceEvaluator(),
            MonetizationEvaluator(
                settings.content_rules,
                min_reply_worthiness=settings.monetization.min_reply_worthiness,
                block_engagement_bait=settings.monetization.block_engagement_bait,
            ),
            SlopDetector(),
            RepetitionDetector(),
            VoiceEvaluator(voice),
        ]

    def context_for(
        self,
        platform: Platform,
        content_type: str,
        *,
        history: list[str] | None = None,
        recent_topics: list[str] | None = None,
        topic: str = "",
        has_verified_experience: bool = False,
    ) -> EvaluationContext:
        return EvaluationContext(
            platform=platform,
            content_type=content_type,
            history=history or [],
            recent_topics=recent_topics or [],
            topic=topic,
            has_verified_experience=has_verified_experience,
            slop_exempt=self.settings.content_rules.is_slop_exempt(content_type),
        )

    def run(self, text: str, context: EvaluationContext) -> Assessment:
        """Evaluate a draft, stopping early on a conclusive failure."""
        assessment = Assessment()
        for evaluator in self.evaluators:
            result = evaluator.evaluate(text, context)
            assessment.results.append(result)
            if result.blocking:
                # No point scoring a draft that is already rejected, and on a
                # 72 piece run those saved calls add up.
                break

        self._apply_thresholds(assessment, context)
        return assessment

    def _apply_thresholds(self, assessment: Assessment, context: EvaluationContext) -> None:
        """Turn scores into blocking failures where they fall short.

        Done after the fact rather than inside each evaluator so that the
        thresholds live in one configurable place and an evaluator stays a
        pure measurement.
        """
        if assessment.blocking:
            return

        checks: list[tuple[str, float | None]] = [
            ("voice_match", self.thresholds.voice_match),
        ]
        for name, floor in checks:
            result = assessment.by_name(name)
            if result and result.score is not None and floor is not None and result.score < floor:
                result.blocking = True
                result.reason = (
                    f"{name} {result.score:.1f} is below the floor of {floor:.1f}: {result.reason}"
                )

        for name, ceiling in (
            ("slop", self.thresholds.max_slop_risk),
            ("repetition", self.thresholds.max_repetition_risk),
        ):
            result = assessment.by_name(name)
            if result and result.risk is not None and not result.risk.at_most(ceiling):
                result.blocking = True
                result.reason = (
                    f"{name} risk is {result.risk.value}, above the ceiling of "
                    f"{ceiling.value}: {result.reason}"
                )


def overall_score(assessment: Assessment) -> float:
    """One number, for sorting and for the workbook.

    An unweighted mean of what was actually measured. Deliberately simple:
    a weighted composite invites tuning the weights until the number looks
    good, which is the opposite of what this is for. The blocking flags carry
    the real decision.
    """
    scores = [r.score for r in assessment.results if r.score is not None]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def worst_risk(assessment: Assessment) -> SlopRisk:
    risks = [r.risk for r in assessment.results if r.risk is not None]
    return min(risks, key=lambda risk: risk.rank) if risks else SlopRisk.LOW
