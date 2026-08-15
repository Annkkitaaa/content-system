"""Evaluation.

Every check a draft passes before it reaches the workbook. The deterministic
ones run first because they are free and catch most failures, so a blocking
result short circuits the expensive ones.
"""

from __future__ import annotations

from contentsys.evaluation.base import (
    Assessment,
    EvaluationContext,
    EvaluationResult,
    Evaluator,
)
from contentsys.evaluation.monetization import MonetizationEvaluator
from contentsys.evaluation.repetition import (
    RepetitionDetector,
    idea_overlap,
    opening_move,
    similarity,
    structure_fingerprint,
)
from contentsys.evaluation.slop import SlopDetector, SlopRules, default_rules
from contentsys.evaluation.suite import (
    EvaluationSuite,
    ExperienceEvaluator,
    VoiceEvaluator,
    overall_score,
    worst_risk,
)

__all__ = [
    "Assessment",
    "EvaluationContext",
    "EvaluationResult",
    "EvaluationSuite",
    "Evaluator",
    "ExperienceEvaluator",
    "MonetizationEvaluator",
    "RepetitionDetector",
    "SlopDetector",
    "SlopRules",
    "VoiceEvaluator",
    "default_rules",
    "idea_overlap",
    "opening_move",
    "overall_score",
    "similarity",
    "structure_fingerprint",
    "worst_risk",
]
