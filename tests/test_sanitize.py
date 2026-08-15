"""The guarantee layer.

These checks are duplicated in the prompt on purpose. The prompt asks; this
enforces. Every test here describes a way a model could quietly produce
something that must never reach a draft.
"""

from __future__ import annotations

import pytest

from contentsys.content.sanitize import (
    EM_DASH,
    EN_DASH,
    HORIZONTAL_BAR,
    find_experience_claims,
    sanitize,
    strip_banned_punctuation,
)


class TestPunctuation:
    @pytest.mark.parametrize("bad", [EM_DASH, HORIZONTAL_BAR])
    def test_banned_dashes_are_replaced(self, bad: str) -> None:
        cleaned, changed = strip_banned_punctuation(f"one thing {bad} then another")

        assert changed
        assert bad not in cleaned
        assert cleaned == "one thing, then another"

    def test_a_spaced_en_dash_is_replaced(self) -> None:
        cleaned, changed = strip_banned_punctuation(f"one thing {EN_DASH} another")

        assert changed
        assert cleaned == "one thing, another"

    def test_an_unspaced_en_dash_survives(self) -> None:
        # A numeric range is legitimate and rewriting it would be wrong.
        text = f"the 2024{EN_DASH}2025 season"

        cleaned, changed = strip_banned_punctuation(text)

        assert not changed
        assert cleaned == text

    def test_no_double_comma_is_left_behind(self) -> None:
        # A naive replacement next to existing punctuation leaves ", ,".
        cleaned, _ = strip_banned_punctuation(f"i thought, {EM_DASH} wrongly, that it held")

        assert ", ," not in cleaned
        assert ",," not in cleaned

    def test_clean_text_is_untouched(self) -> None:
        text = "nothing wrong here, at all."

        cleaned, changed = strip_banned_punctuation(text)

        assert not changed
        assert cleaned == text

    def test_existing_double_spaces_survive(self) -> None:
        # This voice double spaces after a full stop. The tidy-up passes exist
        # to clean up after this function's own replacements, and letting them
        # run over text that never had a dash destroys a real habit.
        text = "one thing.  then another."

        cleaned, changed = strip_banned_punctuation(text)

        assert not changed
        assert cleaned == text

    def test_double_spaces_survive_alongside_a_real_replacement(self) -> None:
        cleaned, changed = strip_banned_punctuation(f"first.  second {EM_DASH} third")

        assert changed
        assert ".  second" in cleaned
        assert EM_DASH not in cleaned


class TestExperienceClaims:
    @pytest.mark.parametrize(
        "text",
        [
            "when i was working on a rollup, this came up constantly",
            "i built a prover once and it was miserable",
            "i spent three months on this before it clicked",
            "my team hit this exact problem",
            "in my last role we shipped something similar",
            "i interned somewhere that used this",
            "a few years ago i wrote something like this",
            "i once tried to explain this at a meetup",
            "i remember when this was considered impossible",
        ],
    )
    def test_autobiographical_claims_are_caught(self, text: str) -> None:
        assert find_experience_claims(text), f"missed a first person claim in: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "i think the reduction is the interesting part",
            "i find it strange that nobody mentions this",
            "i keep coming back to how simple the trick is",
            "one thing i appreciate is how much effort goes into reducing a problem",
            "i used to think proving meant checking everything",
            "sumcheck is the workhorse of modern zk",
        ],
    )
    def test_present_tense_thinking_is_not_a_claim(self, text: str) -> None:
        # Opinions and reactions are always safe. Flagging them would make the
        # check unusable, since almost every good post contains one.
        assert not find_experience_claims(text), f"false positive on: {text}"

    def test_invented_metrics_are_caught(self) -> None:
        # A fabricated number is the most damaging invention because it is the
        # most quotable.
        assert find_experience_claims("i cut proving time by 40% with this")


class TestSanitize:
    def test_a_clean_draft_passes(self) -> None:
        result = sanitize("i keep coming back to how much of this is just reduction.")

        assert result.ok
        assert not result.changed

    def test_an_em_dash_is_repaired_not_rejected(self) -> None:
        # Punctuation is a wording problem, so it is fixed rather than failed.
        result = sanitize(f"the trick is simple {EM_DASH} you never check directly")

        assert result.ok
        assert EM_DASH not in result.text
        assert "replaced em dashes" in result.repairs

    def test_an_invented_experience_is_rejected_not_repaired(self) -> None:
        # Rewriting this would launder the invention into vaguer language,
        # which is worse than failing: it hides the problem.
        result = sanitize("when i was working at a protocol team, we hit this constantly")

        assert not result.ok
        assert "not in the knowledge base" in result.violations[0]

    def test_the_same_claim_passes_when_it_is_backed(self) -> None:
        result = sanitize(
            "when i was working through the spartan paper, this is where it clicked",
            has_verified_experience=True,
        )

        assert result.ok

    def test_wrapper_quotes_are_removed(self) -> None:
        result = sanitize('"i think the reduction is the whole point."')

        assert not result.text.startswith('"')
        assert "removed wrapper quotes" in result.repairs

    def test_internal_quotes_survive(self) -> None:
        text = 'every rabbit hole starts with "just five minutes"'

        assert sanitize(text).text == text

    def test_length_limit_is_enforced(self) -> None:
        result = sanitize("x" * 300, max_length=280)

        assert not result.ok
        assert "too long" in result.violations[0]

    def test_empty_output_is_rejected(self) -> None:
        result = sanitize("   ")

        assert not result.ok
        assert "empty" in result.violations[0]

    def test_habits_worth_keeping_are_kept(self) -> None:
        # Double spacing and stretched letters are real tells in this voice.
        # A sanitiser that flattens them is destroying the thing it protects.
        text = "ugggh.  this took way too long."

        result = sanitize(text)

        assert "ugggh" in result.text
        assert ".  this" in result.text

    def test_excess_blank_lines_are_collapsed(self) -> None:
        result = sanitize("first line\n\n\n\n\nsecond line")

        assert result.text == "first line\n\nsecond line"
