"""The evaluation suite, including the labelled regression corpus.

The corpus tests are the ones that matter most. Adding a slop pattern that
starts rejecting the owner's real posts should break the build rather than
ship, and without a labelled set of authentic examples nothing would notice.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from contentsys.config import Platform, Settings, SlopRisk, get_settings
from contentsys.evaluation import (
    EvaluationContext,
    EvaluationSuite,
    ExperienceEvaluator,
    MonetizationEvaluator,
    RepetitionDetector,
    SlopDetector,
    idea_overlap,
    opening_move,
    overall_score,
    similarity,
    structure_fingerprint,
)
from contentsys.voice.surface import analyse

CORPUS = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "corpus.yaml").read_text(encoding="utf-8")
)

REAL_POSTS = [item["text"] for item in CORPUS["authentic"]]


def texts(label: str) -> list[str]:
    return [item["text"].strip() for item in CORPUS[label]]


def labelled(label: str) -> list[tuple[str, str]]:
    return [(item["text"].strip(), item.get("note", "")) for item in CORPUS[label]]


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return Settings()


@pytest.fixture
def suite(settings: Settings) -> EvaluationSuite:
    return EvaluationSuite(settings, analyse(REAL_POSTS))


def ctx(content_type: str = "technical", **overrides) -> EvaluationContext:
    base = {"platform": Platform.X, "content_type": content_type}
    return EvaluationContext(**{**base, **overrides})


class TestSlopDetector:
    @pytest.mark.parametrize(("text", "note"), labelled("slop"))
    def test_slop_is_caught(self, text: str, note: str) -> None:
        result = SlopDetector().evaluate(text, ctx())

        assert result.risk is SlopRisk.HIGH, f"missed slop ({note}): {result.reason}"

    @pytest.mark.parametrize(("text", "note"), labelled("authentic"))
    def test_real_posts_are_not_flagged(self, text: str, note: str) -> None:
        # The regression that matters. A new pattern that starts rejecting the
        # owner's own writing must break the build.
        result = SlopDetector().evaluate(text, ctx())

        assert result.risk is SlopRisk.LOW, (
            f"false positive on a real post ({note}): {result.reason}"
        )

    @pytest.mark.parametrize(("text", "note"), labelled("good"))
    def test_good_writing_passes(self, text: str, note: str) -> None:
        assert SlopDetector().evaluate(text, ctx()).risk is SlopRisk.LOW

    def test_exempt_content_types_skip_the_check(self) -> None:
        # Reflection reads as generic to a detector precisely because the
        # genre is generic. Exempting it is honest; weakening the detector for
        # everything would not be.
        slop = texts("slop")[1]

        result = SlopDetector().evaluate(slop, ctx("personal_reflection", slop_exempt=True))

        assert result.risk is SlopRisk.LOW
        assert result.details["exempt"]

    def test_the_penalty_is_reported(self) -> None:
        result = SlopDetector().evaluate(texts("slop")[0], ctx())

        assert result.details["penalty"] > 0
        assert result.details["hits"]

    def test_a_hook_in_a_later_paragraph_is_caught(self) -> None:
        # Slop stacks manufactured hooks, so the second paragraph opens on one
        # too. Checking only the very first sentence missed that.
        text = "Some ordinary opening line here.\n\nMost people don't understand this."

        result = SlopDetector().evaluate(text, ctx())

        assert result.details["penalty"] > 0
        assert any("most people" in hit for hit in result.details["hits"])

    def test_a_pattern_mid_sentence_is_not_an_opening(self) -> None:
        # Anchoring to paragraph starts is what keeps this from firing on
        # innocent prose that happens to contain the words.
        result = SlopDetector().evaluate(
            "the proof is short because most people don't need the full transcript", ctx()
        )

        assert not any("opens on" in hit for hit in result.details["hits"])

    def test_shouting_is_penalised_but_acronyms_are_not(self) -> None:
        shouted = SlopDetector().evaluate("This is TRULY AMAZING work", ctx())
        acronyms = SlopDetector().evaluate("R1CS and SNARK and FRI all matter here", ctx())

        assert shouted.details["penalty"] > acronyms.details["penalty"]

    def test_rules_come_from_config_not_source(self, tmp_path: Path) -> None:
        # What reads as machine written moves, so the list has to be editable
        # without touching code.
        from contentsys.evaluation.slop import SlopRules

        path = tmp_path / "rules.yaml"
        path.write_text(
            "bands: {high: 1.0, medium: 0.5}\n"
            "phrases: [{pattern: 'banana', weight: 2.0}]\n"
            "openings: []\nstructures: {}\npunctuation: {}\n",
            encoding="utf-8",
        )

        detector = SlopDetector(SlopRules.load(path))

        assert detector.evaluate("banana bread", ctx()).risk is SlopRisk.HIGH
        assert detector.evaluate("game changer", ctx()).risk is SlopRisk.LOW


class TestRepetition:
    def test_an_exact_repeat_is_caught(self) -> None:
        post = "sumcheck collapses an exponential sum to one point."

        result = RepetitionDetector().evaluate(post, ctx(history=[post]))

        assert result.risk is SlopRisk.HIGH
        assert "exact" in result.reason

    def test_case_and_spacing_do_not_hide_a_repeat(self) -> None:
        result = RepetitionDetector().evaluate(
            "Sumcheck  collapses an exponential sum.",
            ctx(history=["sumcheck collapses an exponential sum."]),
        )

        assert result.risk is SlopRisk.HIGH

    def test_a_reworded_repeat_is_caught(self) -> None:
        result = RepetitionDetector().evaluate(
            "sumcheck collapses an exponential sum into a single point",
            ctx(history=["sumcheck collapses an exponential sum to a single point"]),
        )

        assert result.risk is SlopRisk.HIGH
        assert "duplicate" in result.reason

    def test_a_genuinely_new_post_passes(self) -> None:
        result = RepetitionDetector().evaluate(
            "polynomial commitments are what stop the prover changing its mind",
            ctx(history=["sumcheck collapses an exponential sum to one point"]),
        )

        assert result.risk is SlopRisk.LOW

    def test_a_repeated_shape_is_caught(self) -> None:
        # Three posts with the same structure read as a formula even when
        # every word differs.
        shape = "i used to think {}. turns out {}."
        history = [
            shape.format("proofs were checks", "they are reductions"),
            shape.format("ffts were needed", "sumcheck replaced them"),
            shape.format("setup was required", "transparency exists"),
        ]

        result = RepetitionDetector().evaluate(
            shape.format("commitments were simple", "they are binding"), ctx(history=history)
        )

        assert result.risk is not SlopRisk.LOW
        assert "structural" in result.reason or "emotional" in result.reason

    def test_a_repeated_opening_move_is_caught(self) -> None:
        history = [f"i realised that {n} is interesting today" for n in range(4)]

        result = RepetitionDetector().evaluate(
            "i realised something else entirely", ctx(history=history)
        )

        assert "emotional" in result.reason

    def test_a_topic_inside_its_cooldown_is_caught(self) -> None:
        result = RepetitionDetector().evaluate(
            "something new about it", ctx(topic="sumcheck", recent_topics=["Sumcheck"])
        )

        assert "topic" in result.reason

    def test_an_empty_history_never_flags(self) -> None:
        assert RepetitionDetector().evaluate("anything at all here", ctx()).risk is SlopRisk.LOW

    def test_similarity_bounds(self) -> None:
        assert similarity("the same text", "the same text") == 1.0
        assert similarity("", "anything") == 0.0

    def test_idea_overlap_catches_a_contained_claim(self) -> None:
        # A short post fully contained in a longer one is the same idea.
        assert (
            idea_overlap(
                "sumcheck replaces ffts",
                "sumcheck replaces ffts entirely and that makes the prover linear",
            )
            == 1.0
        )

    def test_opening_moves_are_classified(self) -> None:
        assert opening_move("i realised something") == "realisation"
        assert opening_move("i used to think proofs were checks") == "used to think"
        assert opening_move("polynomial commitments are binding") is None

    def test_structure_fingerprints_match_on_shape_not_words(self) -> None:
        a = structure_fingerprint("i used to think one thing. turns out another.")
        b = structure_fingerprint("i used to think something else. turns out differently.")

        assert a == b


class TestMonetization:
    @pytest.fixture
    def evaluator(self, settings: Settings) -> MonetizationEvaluator:
        return MonetizationEvaluator(settings.content_rules)

    @pytest.mark.parametrize(("text", "note"), labelled("bait"))
    def test_bait_blocks_on_x(self, evaluator: MonetizationEvaluator, text: str, note: str) -> None:
        # A program violation, not a quality problem. Averaging it into a
        # composite score would let a well written violation through.
        result = evaluator.evaluate(text, ctx())

        assert result.blocking, f"bait was not blocked ({note})"
        assert "violation" in result.reason

    def test_the_same_wording_is_fine_on_linkedin(self, evaluator: MonetizationEvaluator) -> None:
        # No such program applies there, and a closing question is normal for
        # the format. One blanket rule would be wrong on both platforms.
        result = evaluator.evaluate(
            "what resources helped you? let me know in the comments",
            ctx(platform=Platform.LINKEDIN, content_type="research_insight"),
        )

        assert not result.blocking

    def test_a_genuine_question_is_not_bait(self, evaluator: MonetizationEvaluator) -> None:
        # The good version and the banned version look similar and are
        # opposites, so this distinction has to hold.
        result = evaluator.evaluate(
            "is threads worth it? like i have not met anyone who uses it. do you?", ctx()
        )

        assert not result.blocking

    def test_concrete_posts_score_higher_than_abstract_ones(
        self, evaluator: MonetizationEvaluator
    ) -> None:
        concrete = evaluator.evaluate(
            "sumcheck sends two coefficients per round, which is why it replaced ffts", ctx()
        )
        abstract = evaluator.evaluate(
            "technology keeps moving forward and that is worth thinking about", ctx()
        )

        assert concrete.score > abstract.score

    def test_conversational_types_are_not_held_to_a_reply_bar(
        self, evaluator: MonetizationEvaluator
    ) -> None:
        # Holding humour to a conversational bar pushes the whole feed toward
        # one register.
        result = evaluator.evaluate("good morning 💜", ctx("personal_reflection"))

        assert not result.blocking

    def test_only_bait_blocks_never_reply_worthiness(
        self, evaluator: MonetizationEvaluator
    ) -> None:
        # An earlier version blocked any X post scoring below the reply
        # worthiness floor, which threw out some of the owner's best writing.
        # An aphorism carries no number and no protocol name, so a keyword
        # heuristic cannot see it, and blocking on that heuristic means
        # optimising the feed toward whatever the heuristic can measure.
        aphorism = "every paper changes what you know. the good ones change how you think."

        result = evaluator.evaluate(aphorism, ctx())

        assert not result.blocking
        assert result.score is not None

    def test_a_low_score_is_still_reported(self, evaluator: MonetizationEvaluator) -> None:
        # Not blocking is not the same as not measuring. The signal still
        # reaches the workbook so it can be judged by a person.
        result = evaluator.evaluate("technology keeps moving forward", ctx())

        assert result.details["below_floor"] is True
        assert result.details["notes"]

    def test_bait_still_blocks_after_that_change(self, evaluator: MonetizationEvaluator) -> None:
        assert evaluator.evaluate("follow for more zk content", ctx()).blocking

    def test_bait_blocking_can_be_switched_off(self, settings: Settings) -> None:
        lenient = MonetizationEvaluator(settings.content_rules, block_engagement_bait=False)

        assert not lenient.evaluate("follow for more zk content", ctx()).blocking


class TestExperienceEvaluator:
    @pytest.mark.parametrize(("text", "note"), labelled("fabricated"))
    def test_fabrication_blocks(self, text: str, note: str) -> None:
        result = ExperienceEvaluator().evaluate(text, ctx())

        assert result.blocking, f"fabrication slipped through ({note})"

    def test_the_same_claim_passes_when_backed(self) -> None:
        result = ExperienceEvaluator().evaluate(
            "when i was working through the spartan paper this clicked",
            ctx(has_verified_experience=True),
        )

        assert not result.blocking

    @pytest.mark.parametrize(("text", "note"), labelled("authentic"))
    def test_real_posts_make_no_false_claim(self, text: str, note: str) -> None:
        assert not ExperienceEvaluator().evaluate(text, ctx()).blocking, note


class TestSuite:
    @pytest.mark.parametrize(("text", "note"), labelled("authentic"))
    def test_the_owners_real_posts_pass_the_whole_suite(
        self, suite: EvaluationSuite, text: str, note: str
    ) -> None:
        # The strongest regression there is. If the suite rejects the writing
        # it was built from, the suite is wrong, not the writing.
        assessment = suite.run(text, suite.context_for(Platform.X, "technical"))

        assert assessment.passed, f"rejected a real post ({note}): {assessment.reasons()}"

    @pytest.mark.parametrize(("text", "note"), labelled("bait"))
    def test_bait_fails_the_suite(self, suite: EvaluationSuite, text: str, note: str) -> None:
        assessment = suite.run(text, suite.context_for(Platform.X, "technical"))

        assert not assessment.passed, note

    @pytest.mark.parametrize(("text", "note"), labelled("fabricated"))
    def test_fabrication_fails_the_suite(
        self, suite: EvaluationSuite, text: str, note: str
    ) -> None:
        assessment = suite.run(text, suite.context_for(Platform.X, "technical"))

        assert not assessment.passed, note

    @pytest.mark.parametrize(("text", "note"), labelled("slop"))
    def test_slop_fails_the_suite(self, suite: EvaluationSuite, text: str, note: str) -> None:
        assessment = suite.run(text, suite.context_for(Platform.X, "technical"))

        assert not assessment.passed, note

    def test_a_blocking_failure_short_circuits_the_rest(self, suite: EvaluationSuite) -> None:
        # Scoring a draft that is already rejected wastes calls, and on a 72
        # piece run that adds up.
        assessment = suite.run(
            "i built a prover last year", suite.context_for(Platform.X, "technical")
        )

        assert len(assessment.results) < len(suite.evaluators)

    def test_failures_are_specific_enough_to_regenerate_against(
        self, suite: EvaluationSuite
    ) -> None:
        # A score with no reason cannot drive a retry.
        assessment = suite.run(
            "agree or disagree? let me know in the comments",
            suite.context_for(Platform.X, "technical"),
        )

        assert assessment.reasons()
        assert all(len(reason) > 20 for reason in assessment.reasons())

    def test_the_summary_reads(self, suite: EvaluationSuite) -> None:
        assessment = suite.run(
            "sumcheck sends two coefficients per round, which is why it replaced ffts",
            suite.context_for(Platform.X, "technical"),
        )

        assert "slop" in assessment.summary()
        assert 0 <= overall_score(assessment) <= 10

    def test_thresholds_come_from_settings(self, settings: Settings) -> None:
        strict = EvaluationSuite(settings, analyse(REAL_POSTS))

        assert strict.thresholds.max_slop_risk is SlopRisk.LOW
