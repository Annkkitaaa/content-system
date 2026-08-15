"""AI slop detection.

Rules live in ``config/slop_rules.yaml`` rather than in this file, because
what reads as machine written moves: a phrase that was a tell two years ago is
ordinary now, and the reverse happens too. A blacklist compiled into source is
a blacklist nobody updates.

This is the deterministic pass. It is free, exact, and runs on every draft. A
model based pass reading the same config can be layered on top for the things
patterns cannot see, but the cheap check catches most of it and never needs a
network call.

Slop is a matter of degree, so this reports a risk band rather than blocking.
Engagement bait, which is a program violation, is handled separately in
:mod:`contentsys.evaluation.monetization` and does block.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from contentsys.config import CONFIG_DIR, Platform, SlopRisk
from contentsys.evaluation.base import EvaluationContext, EvaluationResult


@dataclass(frozen=True)
class Rule:
    pattern: str
    weight: float


@dataclass
class SlopRules:
    high: float
    medium: float
    phrases: list[Rule]
    openings: list[Rule]
    structures: dict[str, dict[str, Any]]
    punctuation: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> SlopRules:
        source = path or (CONFIG_DIR / "slop_rules.yaml")
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        bands = data.get("bands", {})
        return cls(
            high=float(bands.get("high", 6.0)),
            medium=float(bands.get("medium", 3.0)),
            phrases=[
                Rule(r["pattern"].lower(), float(r["weight"])) for r in data.get("phrases", [])
            ],
            openings=[
                Rule(r["pattern"].lower(), float(r["weight"])) for r in data.get("openings", [])
            ],
            structures=data.get("structures", {}),
            punctuation=data.get("punctuation", {}),
        )

    def band(self, penalty: float) -> SlopRisk:
        if penalty >= self.high:
            return SlopRisk.HIGH
        if penalty >= self.medium:
            return SlopRisk.MEDIUM
        return SlopRisk.LOW

    def score(self, penalty: float) -> float:
        """Map penalty points onto the 0 to 10 scale the thresholds use.

        Clamped rather than unbounded so one very bad draft cannot drag an
        average into meaninglessness.
        """
        return max(0.0, min(10.0, 10.0 - penalty * 1.4))


@functools.lru_cache(maxsize=1)
def default_rules() -> SlopRules:
    return SlopRules.load()


_NUMBERED_LESSON = re.compile(r"^\s*\d+[.)]\s+\S", re.M)
_BULLET = re.compile(r"^\s*[-*•]\s+\S", re.M)
_EMOJI_BULLET = re.compile(r"^\s*[\U0001F300-\U0001FAFF☀-➿]\s*\S", re.M)
_ALL_CAPS = re.compile(r"\b[A-Z]{4,}\b")
_HASHTAG = re.compile(r"#\w+")
_LESSON_WORD = re.compile(r"\b(lessons?|takeaways?|things i learned|rules?)\b", re.I)


def _opening_sentences(text: str) -> list[str]:
    """The first sentence of the post and of each paragraph.

    Checking only the very first sentence missed the common case: slop stacks
    manufactured hooks, so the second paragraph opens on one too. A pattern
    matched mid-sentence is usually innocent, which is why this is anchored to
    paragraph starts rather than run over the whole text.
    """
    openings: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        stripped = block.strip()
        if not stripped:
            continue
        match = re.search(r"[.!?\n]", stripped)
        first = (stripped[: match.start()] if match else stripped).strip().lower()
        if first:
            openings.append(first)
    return openings


def _structural_penalties(
    text: str, rules: SlopRules, platform: Platform
) -> list[tuple[str, float, str]]:
    """Shape based tells, as (name, weight, explanation)."""
    found: list[tuple[str, float, str]] = []

    def weight_for(name: str) -> float:
        return float(rules.structures.get(name, {}).get("weight", 0.0))

    numbered = len(_NUMBERED_LESSON.findall(text))
    if numbered >= 3 and _LESSON_WORD.search(text):
        found.append(
            (
                "numbered_lesson_list",
                weight_for("numbered_lesson_list"),
                "reads as a numbered list of lessons",
            )
        )

    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip() and not lines[1].strip():
        opener = lines[0].strip()
        # A short opening line followed by a gap is the classic manufactured
        # hook. A long first line is just a paragraph.
        if len(opener.split()) <= 9 and len(text.split()) > 25:
            found.append(
                ("hook_then_gap", weight_for("hook_then_gap"), "opens with a hook and a line break")
            )

    if _EMOJI_BULLET.search(text):
        found.append(("emoji_bullets", weight_for("emoji_bullets"), "uses emoji as bullet points"))

    bullets = len(_BULLET.findall(text))
    if bullets > 4 and len(text.split()) < 160:
        found.append(
            (
                "excessive_bullets",
                weight_for("excessive_bullets"),
                f"{bullets} bullet points in a short post",
            )
        )

    sentences = [s.strip().lower() for s in re.split(r"[.!?]\s+", text.strip()) if s.strip()]
    if len(sentences) >= 4:
        opening_words = set(sentences[0].split())
        closing_words = set(sentences[-1].split())
        # Restating the opening in the closing is filler wearing the shape of
        # a conclusion.
        if opening_words and len(opening_words & closing_words) / len(opening_words) > 0.6:
            found.append(
                (
                    "restated_ending",
                    weight_for("restated_ending"),
                    "the closing line restates the opening",
                )
            )

    punctuation = rules.punctuation
    exclamations = text.count("!")
    limit = int(punctuation.get("max_exclamations", 2))
    if exclamations > limit:
        found.append(
            (
                "exclamations",
                float(punctuation.get("exclamation_weight", 1.0)) * (exclamations - limit),
                f"{exclamations} exclamation marks",
            )
        )

    if platform is Platform.X:
        hashtags = len(_HASHTAG.findall(text))
        hash_limit = int(punctuation.get("max_hashtags_x", 2))
        if hashtags > hash_limit:
            found.append(
                (
                    "hashtags",
                    float(punctuation.get("hashtag_weight", 1.0)) * (hashtags - hash_limit),
                    f"{hashtags} hashtags",
                )
            )

    shouted = [
        word for word in _ALL_CAPS.findall(text) if word not in {"ZK", "R1CS", "SNARK", "FRI"}
    ]
    if shouted:
        found.append(
            (
                "all_caps",
                float(punctuation.get("all_caps_word_weight", 1.5)) * len(shouted),
                f"shouts in capitals: {', '.join(shouted[:3])}",
            )
        )

    return [(name, weight, why) for name, weight, why in found if weight > 0]


class SlopDetector:
    """The deterministic slop pass."""

    name = "slop"

    def __init__(self, rules: SlopRules | None = None) -> None:
        self.rules = rules or default_rules()

    def evaluate(self, text: str, context: EvaluationContext) -> EvaluationResult:
        if context.slop_exempt:
            # Reflection posts read as generic to a detector precisely because
            # the genre is generic. Exempting them is honest; weakening the
            # detector for everything would not be.
            return EvaluationResult(
                evaluator=self.name,
                score=10.0,
                risk=SlopRisk.LOW,
                reason=f"{context.content_type} is exempt from slop scoring",
                details={"exempt": True},
            )

        lowered = text.lower()
        penalty = 0.0
        hits: list[str] = []

        for rule in self.rules.phrases:
            occurrences = lowered.count(rule.pattern)
            if occurrences:
                penalty += rule.weight * occurrences
                hits.append(f'"{rule.pattern}"')

        for opening in _opening_sentences(text):
            for rule in self.rules.openings:
                if opening.startswith(rule.pattern):
                    penalty += rule.weight
                    hits.append(f'opens on "{rule.pattern}"')

        for _, weight, why in _structural_penalties(text, self.rules, context.platform):
            penalty += weight
            hits.append(why)

        risk = self.rules.band(penalty)
        reason = (
            "reads as machine written: " + "; ".join(hits[:5]) if hits else "no slop patterns found"
        )
        return EvaluationResult(
            evaluator=self.name,
            score=self.rules.score(penalty),
            risk=risk,
            reason=reason,
            details={"penalty": round(penalty, 2), "hits": hits},
        )
