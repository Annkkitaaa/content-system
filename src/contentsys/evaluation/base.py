"""What an evaluator returns.

One shape for every check, so the suite can aggregate them without knowing
what any individual one does, and so a new evaluator is a new file rather than
a change to the pipeline.

The distinction that matters is ``blocking``. Most evaluators produce a score
that gets weighed against a threshold, and a draft that scores badly is
regenerated. A blocking evaluator is different: it fails the draft outright
regardless of how everything else scored. Engagement bait is the case that
forced this. Under the X Original Content Rewards program it is a violation,
not a quality problem, and averaging it into a composite score would let a
well written violation through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from contentsys.config import Platform, SlopRisk


@dataclass
class EvaluationResult:
    """One evaluator's verdict on one draft."""

    evaluator: str
    #: 0 to 10, higher is better. None when the evaluator is a pass or fail
    #: check rather than a graded one.
    score: float | None = None
    #: A risk band, for the evaluators that report one rather than a score.
    risk: SlopRisk | None = None
    #: True when this alone rejects the draft, whatever else passed.
    blocking: bool = False
    #: Human readable, and specific enough to feed back into a retry. A score
    #: with no reason cannot drive regeneration.
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.blocking


@dataclass
class Assessment:
    """Every evaluator's verdict on one draft, plus the overall call."""

    results: list[EvaluationResult] = field(default_factory=list)

    def by_name(self, name: str) -> EvaluationResult | None:
        return next((r for r in self.results if r.evaluator == name), None)

    def score(self, name: str) -> float | None:
        result = self.by_name(name)
        return result.score if result else None

    @property
    def blocking(self) -> list[EvaluationResult]:
        return [result for result in self.results if result.blocking]

    @property
    def passed(self) -> bool:
        return not self.blocking

    def reasons(self) -> list[str]:
        """Why this draft failed, specific enough to regenerate against."""
        return [result.reason for result in self.blocking if result.reason]

    def summary(self) -> str:
        parts = []
        for result in self.results:
            if result.score is not None:
                parts.append(f"{result.evaluator} {result.score:.1f}")
            elif result.risk is not None:
                parts.append(f"{result.evaluator} {result.risk.value}")
        return ", ".join(parts)


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can judge a draft.

    Deliberately narrow. An evaluator gets the text and enough context to
    judge it, and returns a verdict. It does not decide whether the draft is
    regenerated, which is the suite's job, and it does not know about other
    evaluators.
    """

    name: str

    def evaluate(self, text: str, context: EvaluationContext) -> EvaluationResult: ...


@dataclass
class EvaluationContext:
    """What an evaluator needs to know beyond the text itself."""

    platform: Platform
    content_type: str
    #: Posts already written, for repetition checks. Most recent first.
    history: list[str] = field(default_factory=list)
    #: Topics already covered recently, keyed to when.
    recent_topics: list[str] = field(default_factory=list)
    topic: str = ""
    #: Set when the draft is backed by a verified experience.
    has_verified_experience: bool = False
    #: True when this content type is exempt from slop scoring. Reflection
    #: posts read as generic to a detector precisely because the genre is.
    slop_exempt: bool = False
