"""Deterministic voice measurement.

Everything here is computed from text with no model call: it is cheap, exact,
reproducible, and testable. That matters because this is the layer used to
*check* generated output, and a checker that is itself a language model can
be talked out of its own findings.

What this layer deliberately does not do is claim to capture voice. Counting
lowercase letters tells you nothing about how someone thinks. The semantic
profile does that work; this one catches the mechanical drift that makes a
draft feel subtly off even when the ideas are right.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

#: Sentence boundaries. Deliberately simple: this text is social posts, not
#: prose with abbreviations, and a heavyweight tokeniser buys nothing here.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9']+")
_CONTRACTION = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_HASHTAG = re.compile(r"#\w+")
_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+|\b\w+\.(?:com|org|xyz|io|dev)\b")
#: Three or more of the same letter in a row, as in "ugh" stretched out.
_ELONGATION = re.compile(r"([A-Za-z])\1{2,}")

#: Words too common to be characteristic of anyone.
_STOPWORDS = frozenset(
    """
    a about all also am an and any are as at be been but by can could did do does
    for from get had has have he her here him his how i if in into is it its just
    like me more most my no not now of on one only or other our out over said same
    she so some such than that the their them then there these they this those to
    too up us was we were what when where which who why will with would you your
    it's i'm don't that's there's you're i've didn't
    """.split()
)


def _is_emoji(char: str) -> bool:
    # Symbol-other covers the emoji blocks; the pictograph ranges catch the
    # rest without pulling in a dependency.
    if unicodedata.category(char) == "So":
        return True
    return 0x1F300 <= ord(char) <= 0x1FAFF


@dataclass(slots=True)
class SurfaceProfile:
    """Measured mechanics of a body of writing.

    Ratios are 0 to 1 unless named otherwise. Every field is directly
    checkable against generated output.
    """

    sample_count: int = 0
    total_words: int = 0

    # Sentence shape
    mean_sentence_words: float = 0.0
    median_sentence_words: float = 0.0
    p90_sentence_words: float = 0.0
    shortest_sentence_words: int = 0
    longest_sentence_words: int = 0
    mean_sentences_per_post: float = 0.0

    # Casing
    lowercase_opener_ratio: float = 0.0
    lowercase_i_ratio: float = 0.0
    all_lowercase_post_ratio: float = 0.0

    # Punctuation and expressiveness
    exclamation_ratio: float = 0.0
    multi_exclamation_ratio: float = 0.0
    question_ratio: float = 0.0
    ellipsis_ratio: float = 0.0
    contraction_per_100_words: float = 0.0
    elongation_ratio: float = 0.0
    double_space_after_period_ratio: float = 0.0

    # Social mechanics
    emoji_ratio: float = 0.0
    hashtag_ratio: float = 0.0
    mention_ratio: float = 0.0
    link_ratio: float = 0.0

    # Structure
    mean_paragraphs_per_post: float = 0.0
    bullet_post_ratio: float = 0.0
    fragment_ratio: float = 0.0

    # Vocabulary
    distinctive_terms: list[str] = field(default_factory=list)
    common_openers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> list[str]:
        """Plain sentences a human can sanity check.

        A profile nobody reads is a profile nobody notices is wrong.
        """
        lines: list[str] = []
        if self.all_lowercase_post_ratio >= 0.6:
            lines.append(
                f"Writes almost entirely in lowercase "
                f"({self.all_lowercase_post_ratio:.0%} of posts have no capitals)."
            )
        elif self.lowercase_opener_ratio >= 0.5:
            lines.append(
                f"Usually opens sentences lowercase ({self.lowercase_opener_ratio:.0%}), "
                "but capitalises when the post is more considered."
            )
        lines.append(
            f"Sentences run {self.median_sentence_words:.0f} words at the median, "
            f"{self.shortest_sentence_words} at the shortest and "
            f"{self.longest_sentence_words} at the longest."
        )
        lines.append(f"About {self.mean_sentences_per_post:.1f} sentences per post.")
        if self.question_ratio >= 0.1:
            lines.append(f"Asks a real question in {self.question_ratio:.0%} of posts.")
        # Report the habit in both directions. Only mentioning the absence of
        # emoji meant a profile could silently go from "never uses emoji" to
        # saying nothing at all, which reads as agreement rather than as a
        # measurement that changed.
        if self.emoji_ratio < 0.05:
            lines.append("Essentially never uses emoji.")
        else:
            lines.append(
                f"Uses emoji in {self.emoji_ratio:.0%} of posts, so stripping them out "
                "would read as a different person."
            )
        if self.contraction_per_100_words >= 3:
            lines.append(
                f"Uses contractions freely ({self.contraction_per_100_words:.1f} per 100 words), "
                "so the register is conversational rather than formal."
            )
        if self.elongation_ratio > 0:
            lines.append(
                f"Occasionally stretches letters for emphasis in {self.elongation_ratio:.0%} "
                "of posts, which is a real tell and should not be smoothed away."
            )
        if self.distinctive_terms:
            lines.append("Recurring vocabulary: " + ", ".join(self.distinctive_terms[:12]) + ".")
        return lines


def _sentences(text: str) -> list[str]:
    parts = (part.strip() for part in _SENTENCE_SPLIT.split(text.replace("\n", " ")))
    return [part for part in parts if part]


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _looks_like_fragment(sentence: str) -> bool:
    """A sentence with no finite verb, roughly.

    Deliberately crude. The goal is a ratio that moves when the writing style
    moves, not a linguistically defensible parse.
    """
    words = _words(sentence.lower())
    if not words or len(words) > 6:
        return False
    verbish = {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "think",
        "thought",
        "know",
        "knew",
        "feel",
        "felt",
        "turns",
        "changes",
    }
    return not any(word in verbish or word.endswith(("ed", "ing", "s")) for word in words)


def analyse(samples: list[str]) -> SurfaceProfile:
    """Measure a body of writing.

    ``samples`` are whole posts, not sentences, because several of the
    measurements (post-level casing, sentences per post) are only meaningful
    at that granularity.
    """
    texts = [text.strip() for text in samples if text and text.strip()]
    profile = SurfaceProfile(sample_count=len(texts))
    if not texts:
        return profile

    sentence_lengths: list[int] = []
    lowercase_openers = 0
    total_openers = 0
    all_lower_posts = 0
    posts_with_exclamation = 0
    posts_with_multi_exclamation = 0
    posts_with_question = 0
    posts_with_ellipsis = 0
    posts_with_emoji = 0
    posts_with_hashtag = 0
    posts_with_mention = 0
    posts_with_link = 0
    posts_with_bullets = 0
    posts_with_elongation = 0
    posts_with_double_space = 0
    sentences_per_post: list[int] = []
    paragraphs_per_post: list[int] = []
    fragments = 0
    total_sentences = 0
    total_words = 0
    total_contractions = 0
    vocabulary: Counter[str] = Counter()
    openers: Counter[str] = Counter()

    for text in texts:
        stripped = _URL.sub("", text)
        words = _words(stripped)
        total_words += len(words)
        total_contractions += len(_CONTRACTION.findall(stripped))

        letters = [char for char in stripped if char.isalpha()]
        if letters and not any(char.isupper() for char in letters):
            all_lower_posts += 1

        if "!" in text:
            posts_with_exclamation += 1
        if "!!" in text:
            posts_with_multi_exclamation += 1
        if "?" in text:
            posts_with_question += 1
        if "..." in text:
            posts_with_ellipsis += 1
        if any(_is_emoji(char) for char in text):
            posts_with_emoji += 1
        if _HASHTAG.search(text):
            posts_with_hashtag += 1
        if _MENTION.search(text):
            posts_with_mention += 1
        if _URL.search(text):
            posts_with_link += 1
        if re.search(r"^\s*[-*•]\s+", text, re.MULTILINE):
            posts_with_bullets += 1
        if _ELONGATION.search(stripped):
            posts_with_elongation += 1
        if re.search(r"[.!?]  +\S", text):
            posts_with_double_space += 1

        paragraphs = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
        paragraphs_per_post.append(max(1, len(paragraphs)))

        sentences = _sentences(stripped)
        sentences_per_post.append(len(sentences))
        total_sentences += len(sentences)

        for sentence in sentences:
            sentence_words = _words(sentence)
            if sentence_words:
                sentence_lengths.append(len(sentence_words))
            first = next((char for char in sentence if char.isalpha()), None)
            if first is not None:
                total_openers += 1
                if first.islower():
                    lowercase_openers += 1
            if _looks_like_fragment(sentence):
                fragments += 1

        if sentences:
            opener = " ".join(_words(sentences[0].lower())[:2])
            if opener:
                openers[opener] += 1

        for word in words:
            lowered = word.lower()
            if lowered not in _STOPWORDS and len(lowered) > 2 and not lowered.isdigit():
                vocabulary[lowered] += 1

    count = len(texts)
    profile.total_words = total_words
    if sentence_lengths:
        ordered = sorted(sentence_lengths)
        profile.mean_sentence_words = round(statistics.fmean(sentence_lengths), 2)
        profile.median_sentence_words = round(statistics.median(sentence_lengths), 2)
        profile.p90_sentence_words = float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))])
        profile.shortest_sentence_words = ordered[0]
        profile.longest_sentence_words = ordered[-1]
    profile.mean_sentences_per_post = round(statistics.fmean(sentences_per_post), 2)
    profile.mean_paragraphs_per_post = round(statistics.fmean(paragraphs_per_post), 2)

    profile.lowercase_opener_ratio = (
        round(lowercase_openers / total_openers, 3) if total_openers else 0.0
    )
    profile.all_lowercase_post_ratio = round(all_lower_posts / count, 3)
    profile.lowercase_i_ratio = _lowercase_i_ratio(texts)

    profile.exclamation_ratio = round(posts_with_exclamation / count, 3)
    profile.multi_exclamation_ratio = round(posts_with_multi_exclamation / count, 3)
    profile.question_ratio = round(posts_with_question / count, 3)
    profile.ellipsis_ratio = round(posts_with_ellipsis / count, 3)
    profile.emoji_ratio = round(posts_with_emoji / count, 3)
    profile.hashtag_ratio = round(posts_with_hashtag / count, 3)
    profile.mention_ratio = round(posts_with_mention / count, 3)
    profile.link_ratio = round(posts_with_link / count, 3)
    profile.bullet_post_ratio = round(posts_with_bullets / count, 3)
    profile.elongation_ratio = round(posts_with_elongation / count, 3)
    profile.double_space_after_period_ratio = round(posts_with_double_space / count, 3)
    profile.contraction_per_100_words = (
        round(total_contractions / total_words * 100, 2) if total_words else 0.0
    )
    profile.fragment_ratio = round(fragments / total_sentences, 3) if total_sentences else 0.0

    profile.distinctive_terms = [term for term, n in vocabulary.most_common(30) if n > 1][:20]
    profile.common_openers = [opener for opener, n in openers.most_common(8) if n > 1]
    return profile


def _lowercase_i_ratio(texts: list[str]) -> float:
    """How often the standalone pronoun is written lowercase.

    A small thing that is unusually diagnostic: someone who writes "i" is
    making a consistent stylistic choice, and a draft that capitalises it
    reads as written by someone else even when everything else matches.
    """
    lower = upper = 0
    for text in texts:
        lower += len(re.findall(r"(?<![A-Za-z'])i(?![A-Za-z'])", text))
        upper += len(re.findall(r"(?<![A-Za-z'])I(?![A-Za-z'])", text))
    total = lower + upper
    return round(lower / total, 3) if total else 0.0


def compare(profile: SurfaceProfile, candidate: str) -> dict[str, Any]:
    """Check one draft against a measured profile.

    Returns the deviations that matter, so a caller can either reject the
    draft or feed the specifics back into a regeneration prompt. Reporting
    "voice match 6.2" without saying what is off is not actionable.
    """
    draft = analyse([candidate])
    issues: list[str] = []

    if profile.all_lowercase_post_ratio >= 0.6 and draft.all_lowercase_post_ratio < 1.0:
        issues.append("capitalised, but this voice writes in lowercase")

    # Count the capital pronoun directly rather than inferring it from a
    # ratio. A post containing no first person pronoun at all also has a
    # lowercase_i_ratio of zero, and treating that as "uses a capital I"
    # flagged a large share of perfectly good drafts.
    if profile.lowercase_i_ratio >= 0.7:
        capitals = len(re.findall(r"(?<![A-Za-z'])I(?![A-Za-z'])", candidate))
        if capitals:
            issues.append("uses a capital I, but this voice writes i lowercase")

    if profile.emoji_ratio < 0.05 and draft.emoji_ratio > 0:
        issues.append("contains emoji, which this voice does not use")
    if profile.hashtag_ratio < 0.1 and draft.hashtag_ratio > 0:
        issues.append("contains hashtags, which this voice rarely uses here")

    # Compare against the top of the person's range, not their median. A
    # single post is a sample of one or two sentences; a corpus median is
    # dragged down by every "good morning" in the archive. Measuring a point
    # sample against that median flagged ordinary drafts as too long and sent
    # them back for regeneration, which costs a real API call every time.
    if draft.median_sentence_words and profile.p90_sentence_words:
        ceiling = profile.p90_sentence_words * 1.25
        if draft.median_sentence_words > ceiling:
            issues.append(
                f"sentences run long ({draft.median_sentence_words:.0f} words median, "
                f"against {profile.p90_sentence_words:.0f} at this voice's 90th percentile)"
            )

    return {"issues": issues, "measured": draft.to_dict(), "passes": not issues}
