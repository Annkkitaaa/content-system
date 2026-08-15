"""Runtime configuration.

Everything tunable lives here or in the YAML files under ``config/``. Scalar
settings and secrets come from the environment (``.env``); anything with
structure (the weekly content mix, the posting schedule) lives in YAML so it
can be edited without touching code or restarting a thought process.

Nothing in this module reads a secret into a tracked file. ``.env`` is
gitignored; ``.env.example`` carries the shape with empty values.
"""

from __future__ import annotations

import functools
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class Platform(StrEnum):
    X = "X"
    LINKEDIN = "LinkedIn"


class ProviderName(StrEnum):
    """Which LLM backend to talk to.

    ``AGENT_SDK`` runs on a Claude subscription and needs no API key.
    ``ANTHROPIC`` uses the Messages API and needs credit on the Console.
    ``MOCK`` is deterministic and offline, for tests and dry runs.
    """

    AGENT_SDK = "agent_sdk"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class SlopRisk(StrEnum):
    """Ordered worst to best so comparisons read naturally."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def rank(self) -> int:
        return {"HIGH": 2, "MEDIUM": 1, "LOW": 0}[self.value]

    def at_most(self, ceiling: SlopRisk) -> bool:
        return self.rank <= ceiling.rank


class DraftStatus(StrEnum):
    IDEA = "Idea"
    DRAFT = "Draft"
    REVIEW = "Review"
    EDITED = "Edited"
    APPROVED = "Approved"
    PUBLISHED = "Published"
    REJECTED = "Rejected"


class Thresholds(BaseModel):
    """A draft must clear all of these or it goes back for regeneration.

    Scores are on a 0 to 10 scale. Defaults come from the brief.
    """

    model_config = {"frozen": True}

    authenticity: float = Field(default=8.0, ge=0, le=10)
    originality: float = Field(default=7.0, ge=0, le=10)
    voice_match: float = Field(default=8.0, ge=0, le=10)
    technical_accuracy: float = Field(default=8.0, ge=0, le=10)
    max_slop_risk: SlopRisk = SlopRisk.LOW
    max_repetition_risk: SlopRisk = SlopRisk.LOW

    #: How many times to regenerate a failing draft before giving up and
    #: flagging it for review rather than silently shipping something weak.
    max_regeneration_attempts: int = Field(default=2, ge=0, le=5)


class MonetizationRules(BaseModel):
    """X Original Content Rewards constraints.

    These are program rules, not style preferences. ``block_engagement_bait``
    defaults to True because engagement-bait calls to action are a program
    violation, so a bait verdict fails a draft outright rather than docking
    its score.
    """

    model_config = {"frozen": True}

    enabled: bool = True
    block_engagement_bait: bool = True

    #: Program eligibility gates, tracked so the workbook can show progress.
    required_verified_followers: int = Field(default=500, ge=0)
    required_verified_impressions_90d: int = Field(default=500_000, ge=0)

    #: Ranking rewards conversation depth, so a draft that gives a reader
    #: nothing specific to reply to is weak even if it is authentic.
    min_reply_worthiness: float = Field(default=5.0, ge=0, le=10)

    #: Topic scattering resets interest-graph classification. Cap how much of
    #: a week may sit outside the core technical universe.
    max_offtopic_share: float = Field(default=0.25, ge=0, le=1)


class PostingWindow(BaseModel):
    """A named time band that posts are spread across."""

    model_config = {"frozen": True}

    name: str
    start: str  # "HH:MM"
    end: str  # "HH:MM"
    weight: float = Field(default=1.0, gt=0)

    @field_validator("start", "end")
    @classmethod
    def _valid_clock(cls, value: str) -> str:
        hours, _, minutes = value.partition(":")
        if not (hours.isdigit() and minutes.isdigit()):
            raise ValueError(f"expected HH:MM, got {value!r}")
        if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59):
            raise ValueError(f"not a valid time of day: {value!r}")
        return value

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start >= self.end:
            raise ValueError(f"window {self.name!r} starts at or after it ends")
        return self


class PlatformSchedule(BaseModel):
    """When and how often to post on one platform."""

    model_config = {"frozen": True}

    posts_per_day: int | None = None
    posts_per_week: int | None = None
    preferred_days: list[str] = Field(default_factory=list)
    windows: list[PostingWindow] = Field(default_factory=list)

    #: Random offset applied to each computed slot. Machine-regular timing is
    #: a demotion signal, so slots are deliberately not on the clock.
    jitter_minutes: int = Field(default=11, ge=0, le=60)

    #: Never place two posts closer together than this.
    min_gap_minutes: int = Field(default=25, ge=0)

    @model_validator(mode="after")
    def _one_cadence(self) -> Self:
        if (self.posts_per_day is None) == (self.posts_per_week is None):
            raise ValueError("set exactly one of posts_per_day or posts_per_week")
        return self

    def weekly_total(self) -> int:
        if self.posts_per_week is not None:
            return self.posts_per_week
        assert self.posts_per_day is not None
        return self.posts_per_day * 7


class ScheduleConfig(BaseModel):
    model_config = {"frozen": True}

    x: PlatformSchedule
    linkedin: PlatformSchedule

    def for_platform(self, platform: Platform) -> PlatformSchedule:
        return self.x if platform is Platform.X else self.linkedin

    def weekly_total(self) -> int:
        return self.x.weekly_total() + self.linkedin.weekly_total()


class ContentMix(BaseModel):
    """Target distribution of content types, per platform.

    Weights are relative, not percentages, so you can add a type without
    rebalancing every other number.
    """

    model_config = {"frozen": True}

    x: dict[str, float]
    linkedin: dict[str, float]

    @field_validator("x", "linkedin")
    @classmethod
    def _positive_weights(cls, value: dict[str, float], info: ValidationInfo) -> dict[str, float]:
        if not value:
            raise ValueError(f"{info.field_name} mix is empty")
        bad = sorted(k for k, v in value.items() if v <= 0)
        if bad:
            raise ValueError(f"{info.field_name} mix has non-positive weights: {bad}")
        return value

    def for_platform(self, platform: Platform) -> dict[str, float]:
        return self.x if platform is Platform.X else self.linkedin

    def allocate(self, platform: Platform, total: int) -> dict[str, int]:
        """Split ``total`` posts across content types by weight.

        Uses largest-remainder so the parts always sum to exactly ``total``,
        which a naive round-and-hope does not guarantee.
        """
        weights = self.for_platform(platform)
        if total <= 0:
            return dict.fromkeys(weights, 0)

        total_weight = sum(weights.values())
        exact = {name: total * weight / total_weight for name, weight in weights.items()}
        counts = {name: int(value) for name, value in exact.items()}

        shortfall = total - sum(counts.values())
        if shortfall:
            # Hand the leftovers to the largest fractional parts. Ties break on
            # name so the allocation is stable across runs.
            ranked = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), name))
            for name in ranked[:shortfall]:
                counts[name] += 1
        return counts


class Settings(BaseSettings):
    """Scalar settings and secrets, from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CONTENTSYS_",
        extra="ignore",
    )

    database_url: str = "sqlite:///data/contentsys.db"
    export_dir: Path = Path("exports")
    config_dir: Path = CONFIG_DIR

    timezone: str = "Asia/Kolkata"

    provider: ProviderName = ProviderName.AGENT_SDK
    generation_model: str = "claude-opus-5"
    evaluation_model: str = "claude-sonnet-5"

    #: Effort for generation calls. Voice fidelity is the whole product, so
    #: this defaults high rather than to the cheapest setting that works.
    generation_effort: str = "high"
    evaluation_effort: str = "low"

    #: Only read by the Messages API provider. The Agent SDK provider uses
    #: your existing Claude Code login and needs nothing here.
    anthropic_api_key: str | None = None

    #: Generate a larger idea pool than needed so the weakest can be dropped.
    idea_oversample: float = Field(default=2.0, ge=1.0, le=5.0)

    thresholds: Thresholds = Field(default_factory=Thresholds)
    monetization: MonetizationRules = Field(default_factory=MonetizationRules)

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @functools.cached_property
    def schedule(self) -> ScheduleConfig:
        return ScheduleConfig.model_validate(_load_yaml(self.config_dir / "schedule.yaml"))

    @functools.cached_property
    def content_mix(self) -> ContentMix:
        return ContentMix.model_validate(_load_yaml(self.config_dir / "content_mix.yaml"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing config file {path}. Copy the defaults from the repo's config/ directory."
        )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def package_version() -> str:
    """Read the version from pyproject so it is declared in exactly one place."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():  # installed as a wheel, no source tree alongside
        return "0.0.0+unknown"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached so config files are read once.

    Call ``get_settings.cache_clear()`` in tests that need a fresh read.
    """
    return Settings()
