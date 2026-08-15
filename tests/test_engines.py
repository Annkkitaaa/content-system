from __future__ import annotations

import pytest
from sqlmodel import Session

from contentsys.config import ContentRules, Platform, Settings, get_settings
from contentsys.content import modes
from contentsys.content.context import build_context
from contentsys.content.generate import MAX_LENGTH, generate_batch, generate_draft
from contentsys.content.ideas import (
    Idea,
    IdeaPool,
    deduplicate,
    enforce_experience_invariant,
    jaccard,
    parse_ideas,
    rank,
    select,
)
from contentsys.db.models import Confidence, Experience, KnowledgeItem, Opinion
from contentsys.knowledge import load_seed
from contentsys.llm.base import LLMRequest, LLMResponse, Usage
from contentsys.llm.mock import MockProvider
from contentsys.prompts import PromptContext
from contentsys.voice import build_profile
from contentsys.voice.surface import analyse

REAL_POSTS = [
    "i used to think proving something meant checking everything.  turns out some of the "
    "smartest proof systems prove everything by checking almost nothing.",
    "every paper changes what you know. the good ones change how you think.",
    "good morning 💜",
    "is threads worth it? like i have not met anyone who uses it. do you?",
]


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return Settings()


@pytest.fixture
def rules(settings: Settings) -> ContentRules:
    return settings.content_rules


@pytest.fixture
def context(rules: ContentRules) -> PromptContext:
    return PromptContext(
        platform=Platform.X,
        voice=analyse(REAL_POSTS),
        rules=rules,
        knowledge=[KnowledgeItem(concept="Sumcheck", domain="zk", depth="deep")],
        opinions=[Opinion(statement="Reduction is the real content.", strength="strong")],
    )


def make_idea(**overrides) -> Idea:
    base = {
        "topic": "sumcheck",
        "angle": "it got simpler as it got more general",
        "why_interesting": "counterintuitive",
        "content_type": "technical",
        "platform": Platform.X,
    }
    return Idea(**{**base, **overrides})


class TestDeduplication:
    def test_identical_angles_collapse(self) -> None:
        pool = deduplicate([make_idea(), make_idea()])

        assert len(pool.usable) == 1
        assert pool.dropped[0].rejected_because == "duplicate angle"

    def test_different_angles_on_one_topic_survive(self) -> None:
        # Ten good ideas about sumcheck should not collapse into one. Only
        # rephrasings of the same claim should.
        pool = deduplicate(
            [
                make_idea(angle="it got simpler as it got more general"),
                make_idea(angle="the verifier only ever checks a single evaluation point"),
                make_idea(angle="soundness comes entirely from schwartz zippel per round"),
            ]
        )

        assert len(pool.usable) == 3

    def test_reworded_duplicates_are_caught(self) -> None:
        pool = deduplicate(
            [
                make_idea(angle="sumcheck replaces ffts entirely and makes the prover linear"),
                make_idea(angle="sumcheck entirely replaces ffts, making the prover linear"),
            ]
        )

        assert len(pool.usable) == 1

    def test_the_first_one_wins(self) -> None:
        # Ranking runs before dedup, so first-wins means best-wins.
        first = make_idea(novelty=9.0)
        pool = deduplicate([first, make_idea(novelty=1.0)])

        assert pool.usable[0] is first

    def test_jaccard_bounds(self) -> None:
        assert jaccard(frozenset("ab"), frozenset("ab")) == 1.0
        assert jaccard(frozenset("ab"), frozenset("cd")) == 0.0
        assert jaccard(frozenset(), frozenset("ab")) == 0.0


class TestExperienceInvariant:
    def test_ideas_needing_absent_autobiography_are_dropped(self, context: PromptContext) -> None:
        # Caught at the idea stage rather than at draft time. Once the request
        # reaches the generator, the likeliest way it obliges is by inventing
        # the material.
        result = enforce_experience_invariant([make_idea(needs_experience=True)], context)

        assert not result[0].usable
        assert "not verified" in result[0].rejected_because

    def test_they_survive_when_a_verified_experience_exists(self, context: PromptContext) -> None:
        context.experiences = [
            Experience(id=3, summary="Read the Spartan paper", confidence=Confidence.STATED)
        ]

        result = enforce_experience_invariant([make_idea(needs_experience=True)], context)

        assert result[0].usable
        assert result[0].experience_id == 3

    def test_an_inferred_experience_does_not_count(self, context: PromptContext) -> None:
        context.experiences = [
            Experience(id=4, summary="Maybe true", confidence=Confidence.INFERRED)
        ]

        result = enforce_experience_invariant([make_idea(needs_experience=True)], context)

        assert not result[0].usable

    def test_ideas_not_needing_experience_are_untouched(self, context: PromptContext) -> None:
        result = enforce_experience_invariant([make_idea(needs_experience=False)], context)

        assert result[0].usable
        assert result[0].experience_id is None


class TestRankAndSelect:
    def test_novelty_leads(self) -> None:
        ordered = rank([make_idea(novelty=3.0), make_idea(novelty=9.0), make_idea(novelty=6.0)])

        assert [idea.novelty for idea in ordered] == [9.0, 6.0, 3.0]

    def test_depth_breaks_a_novelty_tie(self) -> None:
        ordered = rank(
            [
                make_idea(novelty=5.0, technical_depth="low"),
                make_idea(novelty=5.0, technical_depth="high"),
            ]
        )

        assert ordered[0].technical_depth == "high"

    def test_ordering_is_stable_across_runs(self) -> None:
        batch = [make_idea(novelty=5.0, topic=name) for name in ("c", "a", "b")]

        assert [i.topic for i in rank(batch)] == [i.topic for i in rank(batch)]

    def test_selection_respects_the_content_mix(self) -> None:
        pool = IdeaPool(
            ideas=[make_idea(content_type="technical", angle=f"angle {n}") for n in range(5)]
            + [make_idea(content_type="humor", angle=f"joke {n}") for n in range(5)]
        )

        chosen = select(pool, {"technical": 2, "humor": 1})

        assert sum(1 for i in chosen if i.content_type == "technical") == 2
        assert sum(1 for i in chosen if i.content_type == "humor") == 1

    def test_a_shortfall_is_filled_rather_than_left_as_a_gap(self) -> None:
        # A week slightly off the target mix beats a week with holes in it.
        pool = IdeaPool(
            ideas=[make_idea(content_type="technical", angle=f"a {n}") for n in range(4)]
        )

        chosen = select(pool, {"technical": 2, "humor": 2})

        assert len(chosen) == 4


class TestParsing:
    def test_malformed_entries_are_skipped_not_fatal(self) -> None:
        # Losing one idea out of forty is cheap. Losing the run is not.
        parsed = parse_ideas(
            {
                "ideas": [
                    {"topic": "a", "angle": "a real angle"},
                    {"topic": "", "angle": "no topic"},
                    "not even a dict",
                    {"angle": "no topic key"},
                ]
            },
            Platform.X,
        )

        assert len(parsed) == 1

    def test_novelty_is_clamped(self) -> None:
        parsed = parse_ideas({"ideas": [{"topic": "a", "angle": "b", "novelty": 99}]}, Platform.X)

        assert parsed[0].novelty == 10.0

    def test_a_non_numeric_novelty_falls_back(self) -> None:
        parsed = parse_ideas(
            {"ideas": [{"topic": "a", "angle": "b", "novelty": "very"}]}, Platform.X
        )

        assert parsed[0].novelty == 5.0

    def test_an_empty_payload_is_not_an_error(self) -> None:
        assert parse_ideas({}, Platform.X) == []


class TestDraftGeneration:
    def test_a_clean_draft_comes_back_clean(
        self, context: PromptContext, settings: Settings
    ) -> None:
        draft = generate_draft(MockProvider(), context, make_idea(), settings)

        assert draft.content
        assert draft.attempts == 1

    def test_length_is_enforced_per_platform(
        self, context: PromptContext, settings: Settings
    ) -> None:
        class TooLong:
            name = "toolong"

            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(text="x" * 400, model="m", usage=Usage())

            def complete_json(self, request: LLMRequest) -> dict:
                return {}

        draft = generate_draft(TooLong(), context, make_idea(), settings)

        assert draft.needs_review
        assert any("too long" in issue for issue in draft.issues)
        assert MAX_LENGTH[Platform.X] == 280

    def test_an_invented_experience_is_caught_end_to_end(
        self, context: PromptContext, settings: Settings
    ) -> None:
        # The invariant, all the way through the engine rather than only in
        # the sanitiser unit tests.
        class Fabricator:
            name = "fabricator"

            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(
                    text="when i was working at a protocol team we hit this constantly",
                    model="m",
                    usage=Usage(),
                )

            def complete_json(self, request: LLMRequest) -> dict:
                return {}

        draft = generate_draft(Fabricator(), context, make_idea(), settings)

        assert draft.needs_review
        assert any("not in the knowledge base" in issue for issue in draft.issues)

    def test_a_refusal_is_not_retried(self, context: PromptContext, settings: Settings) -> None:
        # A refusal fails identically on retry, so looping burns quota.
        from contentsys.llm.base import LLMRefusal

        calls = []

        class Refuser:
            name = "refuser"

            def complete(self, request: LLMRequest) -> LLMResponse:
                calls.append(request)
                raise LLMRefusal("declined")

            def complete_json(self, request: LLMRequest) -> dict:
                return {}

        draft = generate_draft(Refuser(), context, make_idea(), settings)

        assert len(calls) == 1
        assert draft.needs_review

    def test_retries_carry_the_specific_failure(
        self, context: PromptContext, settings: Settings
    ) -> None:
        # Regenerating with an unchanged prompt is just rolling the dice again.
        prompts: list[str] = []

        class Failing:
            name = "failing"

            def complete(self, request: LLMRequest) -> LLMResponse:
                prompts.append(request.prompt)
                return LLMResponse(text="x" * 400, model="m", usage=Usage())

            def complete_json(self, request: LLMRequest) -> dict:
                return {}

        generate_draft(Failing(), context, make_idea(), settings)

        assert len(prompts) > 1
        assert "too long" in prompts[1]

    def test_attempts_are_bounded(self, context: PromptContext, settings: Settings) -> None:
        calls = []

        class Failing:
            name = "failing"

            def complete(self, request: LLMRequest) -> LLMResponse:
                calls.append(1)
                return LLMResponse(text="x" * 400, model="m", usage=Usage())

            def complete_json(self, request: LLMRequest) -> dict:
                return {}

        generate_draft(Failing(), context, make_idea(), settings)

        assert len(calls) == settings.thresholds.max_regeneration_attempts + 1

    def test_a_batch_sees_what_it_already_wrote(
        self, context: PromptContext, settings: Settings
    ) -> None:
        # Without this a single run happily writes the same post five times.
        provider = MockProvider()
        ideas = [make_idea(angle=f"angle number {n}") for n in range(3)]

        generate_batch(provider, context, ideas, settings)

        last_prompt = "\n".join(b.text for b in provider.calls[-1].system)
        assert "Recently published" in last_prompt


class TestModes:
    def test_personal_refuses_rather_than_inventing(
        self, context: PromptContext, settings: Settings
    ) -> None:
        # The mode where the invariant is most likely to be tested. It fails
        # loudly instead of producing something plausible.
        context.experiences = []

        draft = modes.personal(MockProvider(), context, settings)

        assert not draft.content
        assert "inventing it" in draft.issues[0]

    def test_personal_works_from_a_verified_experience(
        self, context: PromptContext, settings: Settings
    ) -> None:
        context.experiences = [
            Experience(id=1, summary="Read the Spartan paper", confidence=Confidence.STATED)
        ]

        draft = modes.personal(MockProvider(), context, settings)

        assert draft.content
        assert draft.experience_id == 1

    def test_personal_ignores_an_unverified_experience(
        self, context: PromptContext, settings: Settings
    ) -> None:
        context.experiences = [
            Experience(id=2, summary="Unconfirmed", confidence=Confidence.INFERRED)
        ]

        draft = modes.personal(MockProvider(), context, settings)

        assert not draft.content

    def test_explain_asks_not_to_break_the_concept(
        self, context: PromptContext, settings: Settings
    ) -> None:
        provider = MockProvider()

        modes.explain(provider, context, settings, concept="sumcheck")

        prompt = provider.calls[-1].prompt
        assert "sumcheck" in prompt
        assert "analogy stops holding" in prompt

    def test_reaction_leads_with_the_read_not_the_news(
        self, context: PromptContext, settings: Settings
    ) -> None:
        provider = MockProvider()

        modes.reaction(provider, context, settings, event="a bridge was drained overnight")

        assert "Lead with the interpretation" in provider.calls[-1].prompt

    def test_research_keeps_source_separate_from_experience(
        self, context: PromptContext, settings: Settings
    ) -> None:
        provider = MockProvider()

        modes.from_research(
            provider, context, settings, title="Spartan", summary="A transparent SNARK."
        )

        prompt = provider.calls[-1].prompt
        assert "not personal experience" in prompt


class TestBrainDump:
    def test_the_original_thought_reaches_the_model(
        self, context: PromptContext, settings: Settings
    ) -> None:
        provider = MockProvider()
        thought = (
            "zk was confusing me for so long\n"
            "then i realised you're not proving the secret itself\n"
            "you're proving that there exists something satisfying the constraints"
        )

        modes.brain_dump(provider, context, settings, text=thought)

        prompt = provider.calls[-1].prompt
        assert "proving that there exists something satisfying the constraints" in prompt

    def test_the_instruction_forbids_improving_it(
        self, context: PromptContext, settings: Settings
    ) -> None:
        # The failure mode here is a competent post with the person removed,
        # and that looks like success unless you compare against the input.
        provider = MockProvider()

        modes.brain_dump(provider, context, settings, text="some messy thought")

        prompt = provider.calls[-1].prompt
        assert "Not to improve it" in prompt
        assert "add a hook" in prompt
        assert "almost unchanged" in prompt

    def test_a_real_experience_in_a_dump_is_not_treated_as_invented(
        self, context: PromptContext, settings: Settings
    ) -> None:
        # The owner's own words about their own life are not a fabrication,
        # so the experience check does not apply to this mode.
        class Echo:
            name = "echo"

            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(
                    text="i spent three weeks on this before it clicked",
                    model="m",
                    usage=Usage(),
                )

            def complete_json(self, request: LLMRequest) -> dict:
                return {}

        draft = modes.brain_dump(Echo(), context, settings, text="i spent three weeks on this")

        assert draft.ok, draft.issues


class TestContextBuilding:
    def test_context_loads_from_the_knowledge_base(
        self, session: Session, settings: Settings
    ) -> None:
        from pathlib import Path

        load_seed(session, Path(__file__).resolve().parent.parent / "seed")
        session.commit()
        build_profile(session, Platform.X)
        session.commit()

        context = build_context(session, Platform.X, settings=settings)

        assert context.knowledge
        assert context.opinions
        assert context.experiences
        assert context.voice.sample_count > 0

    def test_low_confidence_preferences_are_withheld(
        self, session: Session, settings: Settings
    ) -> None:
        # One unusual edit should not permanently reshape how everything is
        # written.
        from contentsys.db.models import VoicePreference

        session.add(VoicePreference(key="a", description="seen once", confidence=1))
        session.add(VoicePreference(key="b", description="seen often", confidence=5))
        session.commit()

        context = build_context(session, Platform.X, settings=settings)

        assert any("seen often" in p for p in context.preferences)
        assert not any("seen once" in p for p in context.preferences)
