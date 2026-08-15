"""Repetition detection.

"Have I said this before" has five different meanings, and a system that only
checks one of them will still produce a week that reads as repetitive.

Exact
    The same words. Caught by a normalised hash and by character trigram
    similarity, which survives light rewording.
Topic
    The same subject again too soon.
Idea
    Different words, same claim. The expensive one, so it runs only against
    near neighbours rather than the whole history.
Structural
    The same shape. Hook, three bullets, turn, conclusion, over and over.
Emotional
    The same opening move. A week of "i realised" is repetitive even when
    every post is about something different.

All five are deterministic, which matters: repetition grows with history, and
an evaluator whose cost grows with the archive is one that quietly gets turned
off.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from contentsys.config import SlopRisk
from contentsys.evaluation.base import EvaluationContext, EvaluationResult

#: Character trigram similarity above this is the same post reworded.
NEAR_DUPLICATE = 0.72
#: Content word overlap above this is the same claim in different words.
SAME_IDEA = 0.55
#: How many recent posts to compare against. Repetition that matters is
#: recent; nobody notices an echo of something from four months ago.
WINDOW = 60

_WORD = re.compile(r"[a-z0-9']+")

_FILLER = frozenset(
    """
    a an and are as at be but by for from how i in into is it its me my of on or
    that the their them then there these they this to was we what when which who
    why with you your not no so if
    """.split()
)

#: Opening moves worth counting. A week of any one of these reads as a formula
#: even when the content differs.
_OPENING_MOVES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("realisation", re.compile(r"^\s*i (?:just )?(?:realised|realized)\b", re.I)),
    ("used to think", re.compile(r"^\s*i used to (?:think|believe)\b", re.I)),
    ("nobody mentions", re.compile(r"^\s*(?:nobody|no one)\b", re.I)),
    ("biggest lesson", re.compile(r"^\s*the (?:biggest|hardest|best)\b", re.I)),
    ("heres why", re.compile(r"^\s*here'?s (?:why|the|what)\b", re.I)),
    ("one thing", re.compile(r"^\s*one thing\b", re.I)),
    ("still not over", re.compile(r"^\s*still\b", re.I)),
    ("question", re.compile(r"^\s*(?:is|are|do|does|why|what|how)\b.*\?", re.I | re.S)),
    ("spent time", re.compile(r"^\s*(?:i )?spent\b", re.I)),
)


def normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def exact_key(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()[:32]


def trigrams(text: str) -> set[str]:
    normalised = normalise(text)
    return {normalised[i : i + 3] for i in range(max(0, len(normalised) - 2))}


def similarity(left: str, right: str) -> float:
    """Character trigram overlap, 0 to 1.

    Character level rather than word level on purpose: it survives a synonym
    swap, which is exactly how a near duplicate usually arrives.
    """
    a, b = trigrams(left), trigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def content_words(text: str) -> frozenset[str]:
    return frozenset(
        word for word in _WORD.findall(text.lower()) if word not in _FILLER and len(word) > 2
    )


def idea_overlap(left: str, right: str) -> float:
    a, b = content_words(left), content_words(right)
    if not a or not b:
        return 0.0
    # Overlap relative to the smaller set, so a short post fully contained in
    # a longer one still registers as the same idea.
    return len(a & b) / min(len(a), len(b))


def opening_move(text: str) -> str | None:
    """Classify how a post opens.

    Returns None when the opening does not match a known formula, which is
    the healthy case.
    """
    stripped = text.strip()
    for name, pattern in _OPENING_MOVES:
        if pattern.match(stripped):
            return name
    return None


def structure_fingerprint(text: str) -> str:
    """A coarse description of the post's shape.

    Deliberately coarse. Two posts with identical shapes are repetitive even
    when every word differs, and a fine grained fingerprint would never match.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    bullets = sum(1 for line in lines if re.match(r"^\s*[-*•\d]", line))
    return ":".join(
        [
            f"p{min(len(lines), 6)}",
            f"s{min(len(sentences), 6)}",
            f"b{min(bullets, 4)}",
            "q" if text.rstrip().endswith("?") else "-",
            opening_move(text) or "-",
        ]
    )


@dataclass
class RepetitionFinding:
    kind: str
    detail: str
    severity: SlopRisk


class RepetitionDetector:
    """All five checks, in one pass over a bounded window."""

    name = "repetition"

    def __init__(self, *, window: int = WINDOW) -> None:
        self.window = window

    def evaluate(self, text: str, context: EvaluationContext) -> EvaluationResult:
        history = context.history[: self.window]
        findings: list[RepetitionFinding] = []

        if not text.strip():
            return EvaluationResult(
                evaluator=self.name, score=0.0, risk=SlopRisk.HIGH, reason="empty draft"
            )

        key = exact_key(text)
        for previous in history:
            if exact_key(previous) == key:
                findings.append(
                    RepetitionFinding("exact", "identical to a previous post", SlopRisk.HIGH)
                )
                break

        if not findings:
            for previous in history:
                score = similarity(text, previous)
                if score >= NEAR_DUPLICATE:
                    findings.append(
                        RepetitionFinding(
                            "near duplicate",
                            f"{score:.0%} similar to: {previous.strip()[:70]}",
                            SlopRisk.HIGH,
                        )
                    )
                    break

        # The idea check runs only against posts that already share vocabulary,
        # so its cost stays bounded as the archive grows.
        if not findings:
            for previous in history:
                if similarity(text, previous) < 0.25:
                    continue
                overlap = idea_overlap(text, previous)
                if overlap >= SAME_IDEA:
                    findings.append(
                        RepetitionFinding(
                            "same idea",
                            f"makes the same point as: {previous.strip()[:70]}",
                            SlopRisk.MEDIUM,
                        )
                    )
                    break

        if context.topic and context.topic.lower() in {
            topic.lower() for topic in context.recent_topics
        }:
            findings.append(
                RepetitionFinding(
                    "topic",
                    f"{context.topic} was covered inside its cooldown window",
                    SlopRisk.MEDIUM,
                )
            )

        shape = structure_fingerprint(text)
        matching_shapes = sum(1 for previous in history if structure_fingerprint(previous) == shape)
        if matching_shapes >= 3:
            findings.append(
                RepetitionFinding(
                    "structural",
                    f"{matching_shapes} recent posts share this shape",
                    SlopRisk.MEDIUM,
                )
            )

        move = opening_move(text)
        if move:
            same_move = sum(1 for previous in history if opening_move(previous) == move)
            if same_move >= 3:
                findings.append(
                    RepetitionFinding(
                        "emotional",
                        f'{same_move} recent posts also open on "{move}"',
                        SlopRisk.MEDIUM,
                    )
                )

        if not findings:
            return EvaluationResult(
                evaluator=self.name,
                score=10.0,
                risk=SlopRisk.LOW,
                reason="nothing repeated",
                details={"shape": shape, "opening": move},
            )

        worst = min(findings, key=lambda f: f.severity.rank).severity
        penalty = sum(3.5 if f.severity is SlopRisk.HIGH else 2.0 for f in findings)
        return EvaluationResult(
            evaluator=self.name,
            score=max(0.0, 10.0 - penalty),
            risk=worst,
            reason="; ".join(f"{f.kind}: {f.detail}" for f in findings),
            details={
                "shape": shape,
                "opening": move,
                "findings": [{"kind": f.kind, "detail": f.detail} for f in findings],
            },
        )
