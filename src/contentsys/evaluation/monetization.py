"""X Original Content Rewards compliance.

This evaluator measures two different things, and only one of them blocks.

**Engagement bait blocks.** Under the program it is a violation rather than a
quality problem, so a well written post carrying one is still a violation and
averaging it into a composite score would let it through.

**Reply worthiness does not block.** The ranking model weights reply depth
heavily, so "does this give a reader something specific to respond to" is a
real signal worth measuring. But it is a keyword heuristic, and an earlier
version of this that blocked below a floor threw out some of the owner's best
writing: an aphorism carries no number and no protocol name, so nothing here
can see it. Blocking on a heuristic that cannot recognise a good aphorism
means steering the feed toward whatever the heuristic can measure, which is
the opposite of what this system is for. The score is still recorded, so a
person can judge it.

The two are kept apart deliberately, because they look similar and are
opposites. A question naming a specific thing to answer is the good version.
A question that exists to harvest a reaction is the banned one.
"""

from __future__ import annotations

import re

from contentsys.config import ContentRules
from contentsys.evaluation.base import EvaluationContext, EvaluationResult

#: Posts shorter than this give a reader nothing to hold on to, so they tend
#: to be scrolled past rather than read. Not a rule, a prior.
THIN_POST_WORDS = 8

#: Concrete markers. A post containing a number, a name, or a mechanism gives
#: someone something specific to reply to; one made of abstractions does not.
_CONCRETE = re.compile(
    r"\b(?:\d+|r1cs|snark|stark|sumcheck|fri|kzg|groth16|spartan|plonk|nova|"
    r"poseidon|circom|halo2|ipa|qap|fft|zk|mle)\b",
    re.I,
)
_HEDGE = re.compile(r"\b(?:maybe|perhaps|somewhat|arguably|kind of|sort of)\b", re.I)


class MonetizationEvaluator:
    """Program compliance, and the signals distribution actually rewards."""

    name = "monetization"

    def __init__(
        self,
        rules: ContentRules,
        *,
        min_reply_worthiness: float = 5.0,
        block_engagement_bait: bool = True,
    ) -> None:
        self.rules = rules
        self.min_reply_worthiness = min_reply_worthiness
        self.block_engagement_bait = block_engagement_bait

    def evaluate(self, text: str, context: EvaluationContext) -> EvaluationResult:
        # LinkedIn has no such program, and the identical closing question is
        # completely normal there. One blanket rule would be wrong on both
        # platforms.
        patterns = self.rules.bait_patterns(context.platform)
        lowered = text.lower()

        matched = [pattern for pattern in patterns if pattern in lowered]
        if matched and self.block_engagement_bait:
            return EvaluationResult(
                evaluator=self.name,
                score=0.0,
                blocking=True,
                reason=(
                    "engagement bait, which is a violation of the X Original Content "
                    f'Rewards program rather than a style problem: "{matched[0]}". '
                    "Remove the call to action. A genuine question is fine."
                ),
                details={"bait": matched},
            )

        reply_worthiness, notes = self._reply_worthiness(text, context)

        # Reply worthiness never blocks, and that is a deliberate correction.
        # An earlier version of this rejected any X post scoring below the
        # floor, which threw out some of the owner's best writing: "every
        # paper changes what you know, the good ones change how you think"
        # carries no number, no protocol name, and nothing a keyword can see,
        # and it is exactly the kind of post this account should publish.
        #
        # Bait is a program violation and blocks. Reply worthiness is a
        # heuristic about distribution, and blocking on a heuristic that
        # cannot recognise a good aphorism means optimising the feed toward
        # whatever the heuristic can measure.
        reason = f"reply worthiness {reply_worthiness:.1f}"
        if notes:
            reason += ". " + "; ".join(notes)

        return EvaluationResult(
            evaluator=self.name,
            score=reply_worthiness,
            blocking=False,
            reason=reason,
            details={
                "reply_worthiness": round(reply_worthiness, 2),
                "provenance": "primary_work",
                "notes": notes,
                "below_floor": reply_worthiness < self.min_reply_worthiness,
            },
        )

    def _reply_worthiness(self, text: str, context: EvaluationContext) -> tuple[float, list[str]]:
        """How much a reader is given to respond to.

        Not a popularity prediction. It measures whether the post contains
        something specific enough that a substantive reply is possible, which
        is what the ranking model rewards and what bait only imitates.
        """
        score = 5.0
        notes: list[str] = []
        words = text.split()

        if len(words) < THIN_POST_WORDS:
            score -= 2.0
            notes.append("very short, so there is little to engage with")

        concrete = len(set(m.group(0).lower() for m in _CONCRETE.finditer(text)))
        if concrete:
            score += min(2.5, concrete * 1.25)
        else:
            score -= 1.0
            notes.append("no concrete detail, name or number to respond to")

        # A claim someone could disagree with is repliable. A statement
        # nobody could dispute is not.
        if re.search(r"\b(?:is|are|means|works|does|comes down to)\b", text, re.I):
            score += 0.75

        if text.rstrip().endswith("?"):
            # A real question is the strongest reply prompt there is, which is
            # exactly why bait imitates it. Bait was already blocked above, so
            # anything reaching here is a genuine one.
            score += 1.5

        if _HEDGE.search(text):
            score -= 0.5
            notes.append("hedged, which softens the thing worth replying to")

        if context.content_type in {"humor", "personal_reflection", "random_thought"}:
            # These are not trying to start a conversation, and holding them to
            # a conversational bar would push the whole feed toward one register.
            score = max(score, self.min_reply_worthiness)

        return max(0.0, min(10.0, score)), notes
