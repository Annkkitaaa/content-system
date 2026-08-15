"""The learning loop.

The design question this whole module answers is how fast to learn. Too slow
and it never adapts; too fast and one unusual edit permanently distorts how
everything is written, which is worse because it is hard to notice and hard to
undo. Most of these tests are about that balance.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from contentsys.config import Platform
from contentsys.db.models import ContentEdit, VoicePreference
from contentsys.voice import (
    ACTIVE_CONFIDENCE,
    active_preferences,
    analyse_edit,
    forget,
    learn_from,
    record_edit,
    word_diff,
)


class TestChangeClassification:
    def test_an_identical_edit_teaches_nothing(self) -> None:
        analysis = analyse_edit("same text", "same text")

        assert not analysis.substantive
        assert analysis.changes == []

    def test_a_trivial_fix_teaches_nothing(self) -> None:
        # A one character typo carries no preference, and treating it as one
        # is how a voice model fills up with noise.
        analysis = analyse_edit(
            "sumcheck collapses an exponentail sum to one point",
            "sumcheck collapses an exponential sum to one point",
        )

        assert not analysis.substantive

    def test_shortening_is_recognised(self) -> None:
        analysis = analyse_edit(
            "this is a fairly long draft that says the same thing several times over "
            "and could clearly be much shorter than it currently is",
            "this could be shorter",
        )

        assert any(c.key == "prefers_shorter" for c in analysis.changes)

    def test_lowercasing_is_recognised(self) -> None:
        analysis = analyse_edit(
            "Sumcheck Collapses An Exponential Sum.", "sumcheck collapses an exponential sum."
        )

        assert any(c.key == "prefers_lowercase" for c in analysis.changes)

    def test_a_cut_opening_hook_is_recognised(self) -> None:
        # The most common edit when a draft opens on a manufactured hook.
        analysis = analyse_edit(
            "Here's why this matters.\n\nsumcheck sends two coefficients per round.",
            "sumcheck sends two coefficients per round.",
        )

        assert any(c.key == "cuts_opening_hook" for c in analysis.changes)

    def test_a_single_line_rewrite_is_not_a_cut_hook(self) -> None:
        # An earlier version fired whenever the first line merely changed,
        # so fixing a typo in a one line post taught "stop opening with a
        # hook". That lesson is wrong and would reach every future prompt.
        analysis = analyse_edit(
            "sumcheck collapses an exponentail sum to one point",
            "sumcheck collapses an exponential sum to one point",
        )

        assert not any(c.key == "cuts_opening_hook" for c in analysis.changes)

    def test_a_full_rewrite_is_not_a_cut_hook(self) -> None:
        # If the body did not survive, this is a rewrite and it says nothing
        # about openings.
        analysis = analyse_edit(
            "Here's why this matters.\n\nsumcheck sends two coefficients per round.",
            "polynomial commitments stop the prover changing its mind",
        )

        assert not any(c.key == "cuts_opening_hook" for c in analysis.changes)

    def test_a_long_first_paragraph_is_not_a_hook(self) -> None:
        # A long opening line is a paragraph, not a hook.
        long_opener = " ".join(["word"] * 20)
        analysis = analyse_edit(
            f"{long_opener}\n\nsumcheck sends two coefficients per round.",
            "sumcheck sends two coefficients per round.",
        )

        assert not any(c.key == "cuts_opening_hook" for c in analysis.changes)

    def test_removed_emoji_is_recognised(self) -> None:
        analysis = analyse_edit("this finally clicked 🎉", "this finally clicked, properly")

        assert any(c.key == "removes_emoji" for c in analysis.changes)

    def test_added_hedging_is_recognised(self) -> None:
        analysis = analyse_edit(
            "sumcheck is the only primitive that matters here",
            "sumcheck is maybe the only primitive that matters here, i think",
        )

        assert any(c.key == "softens_claims" for c in analysis.changes)

    def test_removed_hedging_is_recognised(self) -> None:
        # The opposite direction means the opposite thing, so both are needed.
        analysis = analyse_edit(
            "sumcheck is maybe possibly the thing that matters, i think",
            "sumcheck is the thing that matters",
        )

        assert any(c.key == "removes_hedging" for c in analysis.changes)

    def test_added_concrete_detail_is_recognised(self) -> None:
        analysis = analyse_edit(
            "the protocol sends very little per round",
            "sumcheck sends 2 coefficients per round",
        )

        assert any(c.key == "adds_concrete_detail" for c in analysis.changes)

    def test_removed_hashtags_is_recognised(self) -> None:
        analysis = analyse_edit("a real point here #zk #crypto", "a real point here, expanded")

        assert any(c.key == "removes_hashtags" for c in analysis.changes)

    def test_one_edit_can_teach_several_things(self) -> None:
        # A post that was shortened and lowercased says two different things.
        analysis = analyse_edit(
            "Here Is A Long And Overly Capitalised Draft That Goes On For Quite A While Indeed",
            "short and lowercase",
        )

        assert len({c.key for c in analysis.changes}) >= 2

    def test_an_unrecognisable_change_is_reported_not_guessed(self) -> None:
        # A wrong lesson learned confidently is worse than no lesson, because
        # it persists and compounds.
        analysis = analyse_edit("alpha beta gamma delta", "epsilon zeta eta theta")

        assert analysis.unclassified
        assert analysis.changes == []

    def test_word_diff_reports_both_directions(self) -> None:
        removed, added = word_diff("the quick brown fox", "the slow brown fox")

        assert "quick" in removed
        assert "slow" in added


class TestVoiceMemory:
    def test_a_first_observation_is_recorded_but_not_used(self, session: Session) -> None:
        # One edit is a data point, not a habit.
        report = learn_from(session, "A Capitalised Draft Here", "a capitalised draft here")
        session.flush()

        assert "prefers_lowercase" in report.learned
        assert report.now_active == []
        assert active_preferences(session) == []

    def test_repetition_makes_a_preference_active(self, session: Session) -> None:
        for _ in range(ACTIVE_CONFIDENCE):
            learn_from(session, "A Capitalised Draft Here", "a capitalised draft here")
            session.flush()

        keys = [p.key for p in active_preferences(session)]

        assert "prefers_lowercase" in keys

    def test_confidence_is_capped(self, session: Session) -> None:
        # An early preference must stay overturnable by later evidence.
        for _ in range(20):
            learn_from(session, "A Capitalised Draft Here", "a capitalised draft here")
            session.flush()

        preference = session.exec(
            select(VoicePreference).where(VoicePreference.key == "prefers_lowercase")
        ).one()

        assert preference.confidence <= 8

    def test_a_contradiction_weakens_rather_than_fights(self, session: Session) -> None:
        # Evidence the other way is still evidence, so the prompt should never
        # end up holding both sides of a contradiction.
        for _ in range(3):
            learn_from(session, "A Capitalised Draft", "a capitalised draft")
            session.flush()
        before = (
            session.exec(select(VoicePreference).where(VoicePreference.key == "prefers_lowercase"))
            .one()
            .confidence
        )

        report = learn_from(session, "a lowercase draft", "A Lowercase Draft")
        session.flush()
        after = (
            session.exec(select(VoicePreference).where(VoicePreference.key == "prefers_lowercase"))
            .one()
            .confidence
        )

        assert "prefers_lowercase" in report.weakened
        assert after == before - 1

    def test_a_reversal_takes_as_long_to_learn_as_the_original(self, session: Session) -> None:
        # Decaying rather than deleting is what makes this true, and it is the
        # property that stops the voice model whipsawing.
        for _ in range(3):
            learn_from(session, "A Capitalised Draft", "a capitalised draft")
            session.flush()

        for _ in range(3):
            learn_from(session, "a lowercase draft", "A Lowercase Draft")
            session.flush()

        keys = [p.key for p in active_preferences(session)]

        assert "prefers_lowercase" not in keys

    def test_nothing_is_learned_from_a_trivial_edit(self, session: Session) -> None:
        report = learn_from(session, "sumcheck is good", "sumcheck is good")
        session.flush()

        assert not report.analysis.substantive
        assert session.exec(select(VoicePreference)).all() == []

    def test_examples_are_kept_for_traceability(self, session: Session) -> None:
        # A learned lesson has to be traceable back to the edit that produced
        # it, or there is no way to tell a good one from a bad one.
        learn_from(
            session,
            "Here's why this matters.\n\nsumcheck sends two coefficients per round.",
            "sumcheck sends two coefficients per round.",
        )
        session.flush()

        preference = session.exec(
            select(VoicePreference).where(VoicePreference.key == "cuts_opening_hook")
        ).one()

        assert preference.examples

    def test_forgetting_works(self, session: Session) -> None:
        # The system will occasionally learn something wrong, and a voice
        # model with no undo is one nobody keeps feeding.
        learn_from(session, "A Capitalised Draft", "a capitalised draft")
        session.flush()

        assert forget(session, "prefers_lowercase")
        session.flush()

        assert session.exec(select(VoicePreference)).all() == []

    def test_forgetting_something_unknown_is_not_an_error(self, session: Session) -> None:
        assert not forget(session, "never_existed")

    def test_the_report_reads(self, session: Session) -> None:
        report = learn_from(session, "A Capitalised Draft Here", "a capitalised draft here")

        assert "new:" in report.describe()


class TestEditRecording:
    def test_an_edit_is_stored_with_its_summary(self, session: Session) -> None:
        edit = record_edit(
            session,
            draft_id=1,
            original="Sumcheck Is Good",
            edited="sumcheck is good",
        )
        session.flush()

        assert edit.change_summary
        assert "lowercase" in edit.change_summary

    def test_learning_is_opt_in(self, session: Session) -> None:
        # Not every edit expresses a preference. Some are just fixing a fact.
        record_edit(session, draft_id=1, original="a", edited="b", learn=False)
        session.flush()

        stored = session.exec(select(ContentEdit)).one()

        assert stored.learn is False


class TestPromptIntegration:
    def test_active_preferences_reach_the_prompt_context(self, session: Session) -> None:
        from contentsys.content.context import build_context

        for _ in range(ACTIVE_CONFIDENCE):
            learn_from(session, "A Capitalised Draft Here", "a capitalised draft here")
            session.flush()
        session.commit()

        context = build_context(session, Platform.X)

        assert any("lowercase" in preference for preference in context.preferences)

    def test_unconfident_preferences_do_not(self, session: Session) -> None:
        from contentsys.content.context import build_context

        learn_from(session, "A Capitalised Draft Here", "a capitalised draft here")
        session.commit()

        context = build_context(session, Platform.X)

        assert not any("lowercase" in preference for preference in context.preferences)


@pytest.mark.parametrize(
    ("original", "edited", "expected"),
    [
        ("Sumcheck Is Good", "sumcheck is good", "prefers_lowercase"),
        ("a point here 🎉", "a point here, expanded", "removes_emoji"),
        ("this is definitely true", "this is maybe true, i think", "softens_claims"),
        ("wow!!! amazing!!!", "that is genuinely interesting", "removes_exclamations"),
    ],
)
def test_classification_table(original: str, edited: str, expected: str) -> None:
    assert any(change.key == expected for change in analyse_edit(original, edited).changes)
