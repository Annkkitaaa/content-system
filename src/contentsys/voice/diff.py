"""Working out what an edit actually changed.

When the owner rewrites a draft, the interesting thing is not the new text. It
is the difference, because the difference is a statement about how they want
to be written for.

Classification is deterministic. Every category here is something two strings
can be compared on directly: casing, length, punctuation, whether a hook was
cut, whether a claim was hedged. That matters because this feeds the prompt,
and a learning loop driven by a model's opinion of what changed would drift
without anything noticing.

Deliberately conservative. A change that cannot be classified is reported as
unclassified rather than guessed at. A wrong lesson learned confidently is
worse than no lesson, since it persists and compounds across every future
draft.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from contentsys.voice.surface import analyse

_WORD = re.compile(r"[A-Za-z0-9']+")
_HEDGE = re.compile(
    r"\b(?:maybe|perhaps|i think|i guess|possibly|somewhat|arguably|"
    r"kind of|sort of|probably|might|seems?)\b",
    re.I,
)
_CONCRETE = re.compile(r"\b\d+\b|\b(?:r1cs|snark|sumcheck|fri|kzg|groth16|spartan|nova)\b", re.I)
_HASHTAG = re.compile(r"#\w+")

#: Below this, a length change is noise rather than an editorial decision.
LENGTH_SHIFT = 0.18


@dataclass(frozen=True)
class Change:
    """One classified difference.

    ``key`` is stable and becomes the identity of a learned preference, so
    observing the same change twice increments one row rather than creating a
    second.
    """

    key: str
    description: str
    #: 1 means do more of this, -1 means do less.
    polarity: int = 1
    evidence: str = ""


@dataclass
class EditAnalysis:
    changes: list[Change] = field(default_factory=list)
    #: True when text changed but nothing recognisable did. Reported rather
    #: than hidden, so a growing count is a signal the classifier needs work.
    unclassified: bool = False
    similarity: float = 0.0

    @property
    def substantive(self) -> bool:
        """Whether this edit is worth learning from at all.

        A one character fix carries no preference, and treating it as one is
        how a voice model fills up with noise.
        """
        return bool(self.changes) and self.similarity < 0.995

    def summary(self) -> str:
        if not self.changes:
            return "no classifiable change"
        return "; ".join(change.description for change in self.changes)


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _hedges(text: str) -> int:
    return len(_HEDGE.findall(text))


def _blocks(text: str) -> list[str]:
    """Lines with content, blank lines dropped."""
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def _cut_an_opening_hook(original: str, edited: str) -> bool:
    """Whether the first line was dropped and the body kept.

    All three conditions are load bearing. Without the multi-line requirement
    this fires on any single line rewrite. Without checking that the body
    survived, it fires on a complete rewrite, which teaches nothing about
    openings. And a long first line is a paragraph rather than a hook, so the
    word limit keeps this pointed at the thing it is named after.
    """
    original_blocks = _blocks(original)
    if len(original_blocks) < 2:
        return False

    opener = original_blocks[0]
    if len(opener.split()) > 12:
        return False

    normalised_edit = " ".join(edited.lower().split())
    if " ".join(opener.lower().split()) in normalised_edit:
        return False

    # The body has to survive, or this is a rewrite rather than a cut opening.
    body = " ".join(original_blocks[1:]).lower().split()
    if not body:
        return False
    kept = sum(1 for word in set(body) if word in normalised_edit)
    return kept / len(set(body)) >= 0.5


def analyse_edit(original: str, edited: str) -> EditAnalysis:
    """Classify what the owner changed.

    Returns every recognised change, so one edit can teach several things: a
    post that was shortened and lowercased says two different things about how
    they want to be written for.
    """
    analysis = EditAnalysis()
    analysis.similarity = difflib.SequenceMatcher(None, original, edited).ratio()

    if original.strip() == edited.strip():
        return analysis

    before = analyse([original])
    after = analyse([edited])

    original_words = _words(original)
    edited_words = _words(edited)

    # Length. Only a real shift counts, and the direction is the lesson.
    if original_words:
        ratio = (len(edited_words) - len(original_words)) / len(original_words)
        if ratio <= -LENGTH_SHIFT:
            analysis.changes.append(
                Change(
                    "prefers_shorter",
                    f"cut the draft by {abs(ratio):.0%}, so aim shorter",
                    polarity=1,
                    evidence=f"{len(original_words)} words to {len(edited_words)}",
                )
            )
        elif ratio >= LENGTH_SHIFT:
            analysis.changes.append(
                Change(
                    "prefers_longer",
                    f"expanded the draft by {ratio:.0%}, so there was room to say more",
                    polarity=1,
                    evidence=f"{len(original_words)} words to {len(edited_words)}",
                )
            )

    # Casing.
    if after.all_lowercase_post_ratio > before.all_lowercase_post_ratio:
        analysis.changes.append(
            Change("prefers_lowercase", "lowercased the draft, so do not capitalise")
        )
    elif after.all_lowercase_post_ratio < before.all_lowercase_post_ratio:
        analysis.changes.append(
            Change(
                "prefers_capitalisation", "added capitalisation, so this register is not lowercase"
            )
        )

    # The opening. Cutting the first line is the most common edit when a draft
    # opens on a manufactured hook, but detecting it needs care: an earlier
    # version fired whenever the first line merely changed, which meant a
    # one line post with a typo fixed taught "stop opening with a hook". That
    # lesson is wrong and it would have reached every future prompt.
    #
    # The real signature is narrower. The original had a separate opening
    # line, that line is gone, and the body after it survived.
    if _cut_an_opening_hook(original, edited):
        analysis.changes.append(
            Change(
                "cuts_opening_hook",
                "removed the opening line, so start on the substance",
                evidence=_first_line(original)[:80],
            )
        )

    # Emoji.
    if after.emoji_ratio > before.emoji_ratio:
        analysis.changes.append(Change("adds_emoji", "added emoji, so they belong here"))
    elif after.emoji_ratio < before.emoji_ratio:
        analysis.changes.append(
            Change("removes_emoji", "removed emoji, so leave them out here", polarity=-1)
        )

    # Hashtags.
    if len(_HASHTAG.findall(edited)) < len(_HASHTAG.findall(original)):
        analysis.changes.append(
            Change("removes_hashtags", "removed hashtags, so leave them out", polarity=-1)
        )

    # Certainty. Both directions matter and they mean opposite things.
    hedge_delta = _hedges(edited) - _hedges(original)
    if hedge_delta > 0:
        analysis.changes.append(
            Change("softens_claims", "hedged the claim, so do not overstate confidence")
        )
    elif hedge_delta < 0:
        analysis.changes.append(
            Change(
                "removes_hedging",
                "cut the hedging, so commit to the claim rather than qualifying it",
                polarity=-1,
            )
        )

    # Concreteness.
    concrete_delta = len(_CONCRETE.findall(edited)) - len(_CONCRETE.findall(original))
    if concrete_delta > 0:
        analysis.changes.append(
            Change("adds_concrete_detail", "added a specific name or number, so be concrete")
        )

    # Punctuation energy.
    if edited.count("!") < original.count("!"):
        analysis.changes.append(
            Change("removes_exclamations", "removed exclamation marks", polarity=-1)
        )

    # Questions.
    if edited.rstrip().endswith("?") and not original.rstrip().endswith("?"):
        analysis.changes.append(Change("adds_question", "turned the ending into a question"))
    elif original.rstrip().endswith("?") and not edited.rstrip().endswith("?"):
        analysis.changes.append(
            Change("removes_closing_question", "cut the closing question", polarity=-1)
        )

    # Contractions.
    if after.contraction_per_100_words > before.contraction_per_100_words + 0.5:
        analysis.changes.append(
            Change(
                "prefers_contractions", "added contractions, so keep the register conversational"
            )
        )

    if not analysis.changes:
        analysis.unclassified = True

    return analysis


def word_diff(original: str, edited: str) -> tuple[list[str], list[str]]:
    """Words removed and words added.

    Used for the evidence attached to a preference, so a learned lesson can
    always be traced back to the edit that produced it.
    """
    matcher = difflib.SequenceMatcher(None, _words(original.lower()), _words(edited.lower()))
    removed: list[str] = []
    added: list[str] = []
    before, after = _words(original.lower()), _words(edited.lower())
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            removed.extend(before[i1:i2])
        if tag in {"replace", "insert"}:
            added.extend(after[j1:j2])
    return removed, added
