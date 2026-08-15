"""The research layer.

Network is never touched here. Fetchers are exercised through their parsing
and filtering, because a test suite that depends on Hacker News being up is a
test suite that fails for reasons unrelated to the code.

The tests that matter most are the boundary ones. A source says what happened
in the world; the knowledge base says what happened to the owner. If those
ever merge, the system starts inventing autobiography out of news, which is
the exact failure the whole three-layer split exists to prevent.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from contentsys.config import Platform
from contentsys.db.models import KnowledgeItem, ResearchSource
from contentsys.research import (
    Finding,
    ResearchConfig,
    deduplicate,
    has_standing,
    rank,
    recent,
    store,
    to_ideas,
)


def finding(title: str, *, days: float = 1.0, weight: float = 10.0, **kwargs) -> Finding:
    return Finding(
        title=title,
        source=kwargs.pop("source", "Hacker News"),
        url=kwargs.pop("url", f"https://example.test/{abs(hash(title))}"),
        published=datetime.now(UTC) - timedelta(days=days),
        weight=weight,
        **kwargs,
    )


KNOWLEDGE = [
    KnowledgeItem(concept="Sumcheck protocol", domain="zk", depth="deep"),
    KnowledgeItem(concept="Polynomial commitments", domain="zk", depth="working"),
    KnowledgeItem(concept="Circom", domain="zk", depth="working"),
    KnowledgeItem(concept="Nova", domain="zk", depth="aware"),
]


class TestRecency:
    def test_stale_items_are_dropped(self) -> None:
        # Reacting to something three months old reads as manufacturing
        # relevance, which is the failure this layer is meant to avoid.
        kept = recent([finding("fresh", days=2), finding("stale", days=40)], max_age_days=10)

        assert [f.title for f in kept] == ["fresh"]

    def test_an_undated_item_is_treated_as_stale(self) -> None:
        undated = Finding(title="no date", source="X")

        assert recent([undated], max_age_days=10) == []


class TestDeduplication:
    def test_the_same_story_from_two_sources_collapses(self) -> None:
        kept = deduplicate(
            [
                finding("SP1 releases v6.4.0", weight=5, source="GitHub"),
                finding("sp1 releases v6.4.0!", weight=40, source="Hacker News"),
            ]
        )

        assert len(kept) == 1

    def test_the_higher_signal_copy_survives(self) -> None:
        kept = deduplicate([finding("same story", weight=3), finding("same story", weight=90)])

        assert kept[0].weight == 90


class TestRanking:
    def test_recency_outweighs_a_busier_older_story(self) -> None:
        ordered = rank(
            [
                finding("old but busy", days=9, weight=500),
                finding("new and quiet", days=0, weight=3),
            ]
        )

        assert ordered[0].title == "new and quiet"

    def test_attention_breaks_a_recency_tie(self) -> None:
        ordered = rank([finding("quiet", days=1, weight=2), finding("busy", days=1, weight=200)])

        assert ordered[0].title == "busy"


class TestStanding:
    def test_a_topic_the_owner_knows_passes(self) -> None:
        assert has_standing(finding("A new sumcheck protocol variant"), KNOWLEDGE)

    def test_a_topic_the_owner_does_not_know_is_rejected(self) -> None:
        # Reacting to a field you have never read is how an account starts
        # performing expertise instead of having it, and careful phrasing
        # downstream does not fix that.
        assert not has_standing(
            finding("Going Dark, and the era of law enforcement hacking"), KNOWLEDGE
        )
        assert not has_standing(finding("Concept drift in malware classifiers"), KNOWLEDGE)

    def test_a_merely_aware_topic_does_not_grant_standing(self) -> None:
        aware_only = [KnowledgeItem(concept="Nova", domain="zk", depth="aware")]

        assert not has_standing(finding("Nova folding scheme improvements"), aware_only)

    def test_the_abstract_counts_not_just_the_title(self) -> None:
        item = finding("A new construction", summary="built on polynomial commitments")

        assert has_standing(item, KNOWLEDGE)

    def test_an_empty_knowledge_base_grants_no_standing(self) -> None:
        assert not has_standing(finding("anything at all"), [])


class TestIdeaConversion:
    def test_a_finding_becomes_an_idea_carrying_its_source(self) -> None:
        ideas = to_ideas([finding("SP1 released v6.4.0", source="GitHub")], Platform.X)

        assert len(ideas) == 1
        assert ideas[0].source
        assert "SP1" in ideas[0].topic

    def test_the_angle_states_this_is_external(self) -> None:
        # The boundary, written into the instruction the model receives.
        angle = to_ideas([finding("something happened")], Platform.X)[0].angle

        assert "external news" in angle
        assert "not as something you did" in angle

    def test_experience_is_forced_off(self) -> None:
        # Nothing the world did is something the owner did. An idea asking for
        # a first person claim here would be asking the generator to invent one.
        for idea in to_ideas([finding("a"), finding("b")], Platform.X):
            assert idea.needs_experience is False
            assert idea.experience_id is None

    def test_fresher_findings_are_more_novel(self) -> None:
        fresh = to_ideas([finding("now", days=0)], Platform.X)[0]
        older = to_ideas([finding("then", days=8)], Platform.X)[0]

        assert fresh.novelty > older.novelty


class TestStorage:
    def test_findings_are_recorded(self, session: Session) -> None:
        added = store(session, [finding("first"), finding("second")])
        session.commit()

        assert added == 2
        assert len(session.exec(select(ResearchSource)).all()) == 2

    def test_a_rerun_does_not_duplicate(self, session: Session) -> None:
        # The Research sheet should stay a record rather than a growing pile.
        items = [finding("same item")]
        store(session, items)
        session.commit()

        assert store(session, items) == 0


class TestConfig:
    def test_the_shipped_config_parses(self) -> None:
        config = ResearchConfig.load()

        assert 0 < config.reactive_share <= 1
        assert config.arxiv["keywords"]
        assert config.github_releases["repositories"]

    def test_the_reactive_share_matches_the_brief(self) -> None:
        # A quarter, deliberately. Past roughly a third the feed reads as
        # commentary on other people's work rather than thinking in public.
        config = ResearchConfig.load()

        assert config.reactive_count(70) == 18
        assert config.reactive_share <= 0.34

    def test_the_count_never_exceeds_the_week(self) -> None:
        config = ResearchConfig(reactive_share=2.0)

        assert config.reactive_count(10) == 10

    def test_a_zero_share_disables_reactive_posts(self) -> None:
        assert ResearchConfig(reactive_share=0.0).reactive_count(70) == 0

    def test_hacker_news_threshold_is_low_enough_to_return_anything(self) -> None:
        # At 5 points this source returned one item a week, which is not a
        # filter, it is an off switch. ZK stories rarely trend on HN.
        config = ResearchConfig.load()

        assert config.hacker_news["min_points"] <= 3


class TestParsing:
    def test_iso_timestamps_with_and_without_zulu(self) -> None:
        from contentsys.research.sources import _parse_iso

        assert _parse_iso("2026-08-11T10:00:00Z") is not None
        assert _parse_iso("2026-08-11T10:00:00+00:00") is not None
        assert _parse_iso("not a date") is None
        assert _parse_iso(None) is None

    def test_a_naive_timestamp_gets_a_timezone(self) -> None:
        from contentsys.research.sources import _parse_iso

        assert _parse_iso("2026-08-11T10:00:00").tzinfo is not None

    @pytest.mark.parametrize(
        ("left", "right"),
        [("SP1 v6.4.0", "sp1  v6.4.0"), ("A Title!", "a title")],
    )
    def test_dedup_keys_ignore_case_and_punctuation(self, left: str, right: str) -> None:
        assert finding(left).key() == finding(right).key()
