from __future__ import annotations

import pytest
from pydantic import ValidationError

from contentsys.config import (
    ContentMix,
    Platform,
    PlatformSchedule,
    PostingWindow,
    ProviderName,
    Settings,
    SlopRisk,
    Thresholds,
    get_settings,
)


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


def test_defaults_are_usable_without_any_env(settings: Settings) -> None:
    # An empty .env has to work, otherwise first run is a config puzzle.
    assert settings.provider is ProviderName.AGENT_SDK
    assert settings.generation_model == "claude-opus-5"
    assert settings.evaluation_model == "claude-sonnet-5"
    assert settings.timezone == "Asia/Kolkata"
    assert settings.tzinfo.key == "Asia/Kolkata"


def test_shipped_config_files_parse(settings: Settings) -> None:
    assert settings.schedule.x.weekly_total() == 70
    assert settings.schedule.linkedin.weekly_total() == 2
    assert settings.schedule.weekly_total() == 72


def test_unknown_timezone_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown timezone"):
        Settings(timezone="Mars/Olympus_Mons")


class TestThresholds:
    def test_slop_risk_orders_worst_to_best(self) -> None:
        assert SlopRisk.LOW.at_most(SlopRisk.LOW)
        assert SlopRisk.LOW.at_most(SlopRisk.HIGH)
        assert not SlopRisk.HIGH.at_most(SlopRisk.LOW)
        assert not SlopRisk.MEDIUM.at_most(SlopRisk.LOW)

    def test_defaults_match_the_brief(self) -> None:
        thresholds = Thresholds()
        assert thresholds.authenticity == 8.0
        assert thresholds.originality == 7.0
        assert thresholds.technical_accuracy == 8.0
        assert thresholds.max_slop_risk is SlopRisk.LOW

    def test_scores_outside_the_scale_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Thresholds(authenticity=11.0)


class TestContentMix:
    def test_allocation_sums_to_the_exact_target(self, settings: Settings) -> None:
        # Largest-remainder, not round-and-hope: 70 posts across 10 weighted
        # types must come to 70, never 69 or 71.
        for total in (7, 10, 33, 70, 71):
            allocation = settings.content_mix.allocate(Platform.X, total)
            assert sum(allocation.values()) == total

    def test_allocation_is_stable_across_runs(self, settings: Settings) -> None:
        first = settings.content_mix.allocate(Platform.X, 70)
        second = settings.content_mix.allocate(Platform.X, 70)
        assert first == second

    def test_heavier_weight_gets_more_posts(self) -> None:
        mix = ContentMix(x={"technical": 3.0, "humor": 1.0}, linkedin={"essay": 1.0})
        allocation = mix.allocate(Platform.X, 40)
        assert allocation == {"technical": 30, "humor": 10}

    def test_zero_total_allocates_nothing(self, settings: Settings) -> None:
        allocation = settings.content_mix.allocate(Platform.LINKEDIN, 0)
        assert set(allocation.values()) == {0}

    def test_non_positive_weights_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-positive"):
            ContentMix(x={"technical": 0.0}, linkedin={"essay": 1.0})

    def test_empty_mix_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            ContentMix(x={}, linkedin={"essay": 1.0})


class TestPostingWindow:
    def test_backwards_window_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="starts at or after it ends"):
            PostingWindow(name="broken", start="18:00", end="09:00")

    @pytest.mark.parametrize("value", ["25:00", "09:61", "morning", "9am", "0900"])
    def test_malformed_clock_is_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            PostingWindow(name="broken", start=value, end="23:00")


class TestPlatformSchedule:
    def test_exactly_one_cadence_is_required(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            PlatformSchedule(posts_per_day=10, posts_per_week=70)
        with pytest.raises(ValidationError, match="exactly one"):
            PlatformSchedule()

    def test_daily_cadence_becomes_a_weekly_total(self) -> None:
        assert PlatformSchedule(posts_per_day=10).weekly_total() == 70

    def test_jitter_defaults_on(self) -> None:
        # Machine-regular timing is a demotion signal, so a schedule that
        # never jitters should be a deliberate choice, not the default.
        assert PlatformSchedule(posts_per_day=10).jitter_minutes > 0


class TestMonetization:
    def test_bait_blocking_defaults_on(self, settings: Settings) -> None:
        # Engagement bait is a program violation, not a style preference.
        assert settings.monetization.block_engagement_bait is True

    def test_eligibility_gates_match_the_program(self, settings: Settings) -> None:
        assert settings.monetization.required_verified_followers == 500
        assert settings.monetization.required_verified_impressions_90d == 500_000
