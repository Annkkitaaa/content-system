"""Two judgement calls the owner made, encoded as config rather than assumed.

Both are tested because both are the kind of rule that gets quietly inverted
by a later refactor and then nobody notices until the output is wrong.
"""

from __future__ import annotations

import pytest

from contentsys.config import ContentRules, Platform, Settings, VisualPolicy, get_settings


@pytest.fixture
def rules() -> ContentRules:
    get_settings.cache_clear()
    return Settings().content_rules


class TestSlopExemptions:
    def test_reflective_posts_are_exempt(self, rules: ContentRules) -> None:
        # The owner genuinely writes these. The genre reads as generic to a
        # detector precisely because the genre is generic, so the exemption is
        # the honest fix rather than weakening the detector globally.
        assert rules.is_slop_exempt("personal_reflection")

    def test_technical_posts_are_not_exempt(self, rules: ContentRules) -> None:
        # The exemption must stay narrow, or it becomes an off switch.
        assert not rules.is_slop_exempt("technical")
        assert not rules.is_slop_exempt("mini_analysis")
        assert not rules.is_slop_exempt("opinion")

    def test_every_exempt_type_is_capped(self, rules: ContentRules) -> None:
        # Exempting a type without capping it just moves the problem: a week
        # of unchecked reflection posts is exactly the failure being avoided.
        for content_type in rules.slop_exempt:
            assert rules.daily_cap(content_type) is not None, (
                f"{content_type} is exempt from slop checks but has no daily cap"
            )

    def test_caps_are_small(self, rules: ContentRules) -> None:
        for content_type in rules.slop_exempt:
            assert rules.daily_cap(content_type) <= 2


class TestVisualPolicy:
    def test_every_linkedin_post_gets_a_diagram(self, rules: ContentRules) -> None:
        policy = rules.visual_policy(Platform.LINKEDIN)

        assert policy.always
        assert rules.wants_visual(Platform.LINKEDIN, "thoughtful_opinion")

    def test_structural_x_posts_are_eligible(self, rules: ContentRules) -> None:
        assert rules.wants_visual(Platform.X, "technical")
        assert rules.wants_visual(Platform.X, "mini_analysis")

    def test_conversational_x_posts_are_not(self, rules: ContentRules) -> None:
        # An image on a one line thought is decoration, and decoration costs a
        # generation call and a review decision.
        assert not rules.wants_visual(Platform.X, "humor")
        assert not rules.wants_visual(Platform.X, "personal_reflection")
        assert not rules.wants_visual(Platform.X, "opinion")

    def test_x_diagrams_are_capped_per_day(self, rules: ContentRules) -> None:
        assert rules.visual_policy(Platform.X).max_per_day == 3

    def test_every_referenced_kind_is_defined(self, rules: ContentRules) -> None:
        for platform in Platform:
            assert rules.visual_policy(platform).default_kind in rules.kinds

    def test_an_empty_policy_allows_nothing(self) -> None:
        assert not VisualPolicy().allows("technical")


class TestBaitPatterns:
    def test_bait_rules_apply_to_x(self, rules: ContentRules) -> None:
        patterns = rules.bait_patterns(Platform.X)

        assert patterns
        assert "let me know in the comments" in patterns

    def test_bait_rules_do_not_apply_to_linkedin(self, rules: ContentRules) -> None:
        # The same closing question is a program violation on X and completely
        # normal on LinkedIn. One blanket rule would be wrong on both.
        assert rules.bait_patterns(Platform.LINKEDIN) == []

    def test_patterns_are_lowercase_for_matching(self, rules: ContentRules) -> None:
        assert all(pattern == pattern.lower() for pattern in rules.bait_patterns(Platform.X))

    def test_the_genuine_question_carve_out_is_documented(self, rules: ContentRules) -> None:
        # Reply depth is what the ranking model rewards, so a real question is
        # encouraged. The distinction has to be written down or it gets lost.
        assert "bait" in rules.allowed_question_note.lower()
        assert len(rules.allowed_question_note) > 80
