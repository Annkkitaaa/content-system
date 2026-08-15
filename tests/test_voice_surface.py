"""The deterministic voice analyser.

Several tests use the owner's actual posts, because a measurement that scores
well on synthetic text and misreads the real thing is worse than useless: it
is confidently wrong in the direction that matters.
"""

from __future__ import annotations

import pytest

from contentsys.voice.surface import SurfaceProfile, analyse, compare

# Real posts. Lowercase, short, conversational, no emoji.
REAL_X_POSTS = [
    "i used to think proving something meant checking everything.  turns out some of the "
    "smartest proof systems prove everything by checking almost nothing.",
    "every paper changes what you know. the good ones change how you think.",
    "i don't think life changes in big moments.  i think it changes in ordinary days that "
    "you don't realize you'll remember.",
    "is threads worth it? like i have not met anyone who uses it. do you?",
    "ugggggggggggggggghhhhhhhhhhhhh",
    "sunshine, good music, and zero overthinking.",
    "lately i've been so negative whenever it comes to myself.",
    'every rabbit hole starts with "i\'m just going to spend five minutes looking this up."',
]


class TestEmptyInput:
    def test_no_samples_gives_an_empty_profile(self) -> None:
        profile = analyse([])

        assert profile.sample_count == 0
        assert profile.median_sentence_words == 0.0

    def test_blank_samples_are_ignored(self) -> None:
        assert analyse(["", "   ", "\n\n"]).sample_count == 0

    def test_describe_never_crashes_on_an_empty_profile(self) -> None:
        assert isinstance(SurfaceProfile().describe(), list)


class TestCasing:
    def test_detects_an_all_lowercase_voice(self) -> None:
        profile = analyse(REAL_X_POSTS)

        assert profile.all_lowercase_post_ratio >= 0.7
        assert profile.lowercase_opener_ratio >= 0.7

    def test_detects_a_capitalised_voice(self) -> None:
        profile = analyse(["This Is Normal Prose.", "So Is This One."])

        assert profile.all_lowercase_post_ratio == 0.0

    def test_lowercase_i_is_measured_separately(self) -> None:
        # Small, but unusually diagnostic. Someone who writes "i" is making a
        # consistent choice, and capitalising it reads as a different person.
        assert analyse(["i think i know", "i said so"]).lowercase_i_ratio == 1.0
        assert analyse(["I think I know"]).lowercase_i_ratio == 0.0

    def test_a_contraction_does_not_count_as_the_pronoun(self) -> None:
        assert analyse(["i've been reading"]).lowercase_i_ratio == 0.0


class TestSentenceShape:
    def test_measures_length_across_the_distribution(self) -> None:
        profile = analyse(REAL_X_POSTS)

        assert 4 <= profile.median_sentence_words <= 14
        assert profile.shortest_sentence_words < profile.longest_sentence_words

    def test_counts_sentences_per_post(self) -> None:
        profile = analyse(["One. Two. Three.", "Just one."])

        assert profile.mean_sentences_per_post == 2.0

    def test_long_form_reads_as_long_form(self) -> None:
        short = analyse(["short one.", "also short."])
        long_form = analyse(
            [
                "Over the past few days I have been reading the Spartan proof system, "
                "and I wanted to understand why it works rather than memorise it. "
                "At first glance it can feel overwhelming, because the paper is full "
                "of multilinear extensions and algebraic reductions."
            ]
        )

        assert long_form.median_sentence_words > short.median_sentence_words


class TestExpressiveness:
    def test_catches_stretched_letters(self) -> None:
        # A real tell in this voice. Smoothing it away loses the person.
        assert analyse(["ugggggggghhhhhh"]).elongation_ratio == 1.0
        assert analyse(["perfectly normal text"]).elongation_ratio == 0.0

    def test_distinguishes_single_and_double_exclamation(self) -> None:
        profile = analyse(["good morning!!", "hello there!"])

        assert profile.exclamation_ratio == 1.0
        assert profile.multi_exclamation_ratio == 0.5

    def test_counts_questions(self) -> None:
        assert analyse(["do you?", "statement."]).question_ratio == 0.5

    def test_measures_contraction_density(self) -> None:
        rich = analyse(["i don't think it's worth it, i've tried"])
        plain = analyse(["i do not think it is worth it"])

        assert rich.contraction_per_100_words > plain.contraction_per_100_words
        assert plain.contraction_per_100_words == 0.0

    def test_notices_double_spacing_after_a_full_stop(self) -> None:
        # Present in the real samples. A small habit, and free to preserve.
        assert analyse(["one thing.  then another."]).double_space_after_period_ratio == 1.0
        assert analyse(["one thing. then another."]).double_space_after_period_ratio == 0.0


class TestSocialMechanics:
    def test_detects_absence_of_emoji(self) -> None:
        assert analyse(REAL_X_POSTS).emoji_ratio == 0.0

    def test_detects_emoji_when_present(self) -> None:
        assert analyse(["shipping this 🚀"]).emoji_ratio == 1.0

    def test_describe_reports_emoji_use_in_both_directions(self) -> None:
        # Only reporting the absence meant a profile could go from asserting
        # "never uses emoji" to saying nothing, which reads as agreement
        # rather than as a measurement that changed.
        without = " ".join(analyse(["plain text", "more plain text"]).describe())
        with_emoji = " ".join(analyse(["good morning 💜", "the vibe 💜💌"]).describe())

        assert "never uses emoji" in without
        assert "emoji" in with_emoji
        assert "never uses emoji" not in with_emoji

    def test_counts_hashtags_and_mentions(self) -> None:
        profile = analyse(["thanks @nethermind #zk", "no tags here"])

        assert profile.hashtag_ratio == 0.5
        assert profile.mention_ratio == 0.5

    def test_urls_are_excluded_from_word_counts(self) -> None:
        # Otherwise a link inflates sentence length and skews the vocabulary.
        with_link = analyse(["join the waitlist http://torbit.xyz"])

        assert "http" not in with_link.distinctive_terms


class TestVocabulary:
    def test_surfaces_recurring_terms_and_skips_stopwords(self) -> None:
        profile = analyse(["sumcheck is elegant", "sumcheck again", "the and of"])

        assert "sumcheck" in profile.distinctive_terms
        assert "the" not in profile.distinctive_terms

    def test_a_term_used_once_is_not_distinctive(self) -> None:
        assert analyse(["polynomial commitments"]).distinctive_terms == []


class TestDescribe:
    def test_reports_a_lowercase_voice_in_plain_words(self) -> None:
        lines = " ".join(analyse(REAL_X_POSTS).describe()).lower()

        assert "lowercase" in lines
        assert "emoji" in lines

    def test_every_line_is_a_readable_sentence(self) -> None:
        # A profile nobody reads is one nobody notices is wrong.
        for line in analyse(REAL_X_POSTS).describe():
            assert line[0].isupper()
            assert line.endswith(".")


class TestCompare:
    @pytest.fixture
    def profile(self) -> SurfaceProfile:
        return analyse(REAL_X_POSTS)

    def test_a_matching_draft_passes(self, profile: SurfaceProfile) -> None:
        result = compare(profile, "i keep coming back to how much of this is just reduction.")

        assert result["passes"], result["issues"]

    def test_capitalisation_is_flagged(self, profile: SurfaceProfile) -> None:
        result = compare(profile, "I Keep Coming Back To This Idea.")

        assert not result["passes"]
        assert any("lowercase" in issue for issue in result["issues"])

    def test_emoji_is_flagged(self, profile: SurfaceProfile) -> None:
        result = compare(profile, "this finally clicked for me 🎉")

        assert any("emoji" in issue for issue in result["issues"])

    def test_an_overlong_sentence_is_flagged(self, profile: SurfaceProfile) -> None:
        result = compare(
            profile,
            "in the rapidly evolving landscape of modern cryptographic research it has "
            "become increasingly apparent that the reduction of computational problems "
            "into progressively simpler algebraic forms represents a foundational shift.",
        )

        assert any("long" in issue for issue in result["issues"])

    def test_issues_say_what_is_wrong(self, profile: SurfaceProfile) -> None:
        # A score without a reason cannot be fed back into regeneration.
        result = compare(profile, "I Am Very Excited To Share This 🚀")

        assert result["issues"]
        assert all(len(issue) > 10 for issue in result["issues"])
