from __future__ import annotations

import pytest
from sqlmodel import Session

from contentsys.config import ContentRules, Platform, ProviderName, Settings, get_settings
from contentsys.db.models import Confidence, Experience, KnowledgeItem, Opinion
from contentsys.llm import LLMRequest, system_prompt
from contentsys.llm.mock import MockProvider
from contentsys.llm.registry import build_provider
from contentsys.prompts import IDEA_SCHEMA, PromptContext, build_system, draft_request, idea_request
from contentsys.voice.surface import analyse

REAL_POSTS = [
    "i used to think proving something meant checking everything.  turns out some of the "
    "smartest proof systems prove everything by checking almost nothing.",
    "every paper changes what you know. the good ones change how you think.",
    "good morning 💜",
    "the moment you feel you are getting somewhere, they change the rules 🥲",
    "is threads worth it? like i have not met anyone who uses it. do you?",
]


@pytest.fixture
def rules() -> ContentRules:
    get_settings.cache_clear()
    return Settings().content_rules


@pytest.fixture
def context(rules: ContentRules) -> PromptContext:
    return PromptContext(
        platform=Platform.X,
        voice=analyse(REAL_POSTS),
        rules=rules,
        content_type="technical",
        knowledge=[KnowledgeItem(concept="Sumcheck protocol", domain="zk", depth="deep")],
        opinions=[
            Opinion(statement="Reduction is the real content of a paper.", strength="strong")
        ],
    )


class TestSystemPrompt:
    def test_the_voice_measurement_reaches_the_prompt(self, context: PromptContext) -> None:
        text = "\n".join(block.text for block in build_system(context))

        assert "lowercase" in text
        assert "emoji" in text

    def test_house_rules_ban_em_dashes_explicitly(self, context: PromptContext) -> None:
        text = "\n".join(block.text for block in build_system(context))

        assert "em dash" in text

    def test_no_verified_experience_is_stated_as_an_exhaustive_absence(
        self, context: PromptContext
    ) -> None:
        # Silence is ambiguous. Saying "none" explicitly is what makes the
        # constraint legible rather than something to infer.
        text = "\n".join(block.text for block in build_system(context))

        assert "Verified experiences available: none" in text

    def test_verified_experiences_are_listed_with_ids(self, context: PromptContext) -> None:
        context.experiences = [
            Experience(id=7, summary="Read the Spartan paper", confidence=Confidence.STATED)
        ]

        text = "\n".join(block.text for block in build_system(context))

        assert "[7] Read the Spartan paper" in text
        assert "exhaustive" in text

    def test_unconfirmed_experiences_are_not_offered(self, context: PromptContext) -> None:
        # An inferred experience is real information but not permission.
        context.experiences = [
            Experience(id=9, summary="Something unconfirmed", confidence=Confidence.INFERRED)
        ]

        text = "\n".join(block.text for block in build_system(context))

        assert "Something unconfirmed" not in text
        assert "Verified experiences available: none" in text

    def test_bait_rules_appear_on_x_only(self, context: PromptContext, rules: ContentRules) -> None:
        x_text = "\n".join(b.text for b in build_system(context))
        context.platform = Platform.LINKEDIN
        linkedin_text = "\n".join(b.text for b in build_system(context))

        assert "program violation" in x_text
        assert "let me know in the comments" in x_text
        assert "let me know in the comments" not in linkedin_text


class TestCaching:
    def test_exactly_one_cache_breakpoint(self, context: PromptContext) -> None:
        blocks = build_system(context)

        assert sum(1 for block in blocks if block.cacheable) == 1

    def test_the_breakpoint_sits_after_the_stable_sections(self, context: PromptContext) -> None:
        blocks = build_system(context)
        index = next(i for i, block in enumerate(blocks) if block.cacheable)

        # Everything up to and including the breakpoint must be reusable
        # across calls, so nothing per-post may appear in it.
        prefix = "\n".join(block.text for block in blocks[: index + 1])
        assert "Content type:" not in prefix
        assert "Verified experiences" not in prefix

    def test_the_cached_prefix_is_identical_across_content_types(
        self, context: PromptContext
    ) -> None:
        # This is the whole point. If a per-post detail leaks into the prefix,
        # nothing fails, the cache just silently stops paying off.
        def prefix_of(content_type: str) -> str:
            context.content_type = content_type
            blocks = build_system(context)
            index = next(i for i, b in enumerate(blocks) if b.cacheable)
            return "\n".join(b.text for b in blocks[: index + 1])

        assert prefix_of("technical") == prefix_of("humor") == prefix_of("opinion")

    def test_the_prefix_is_the_bulk_of_the_prompt(self, context: PromptContext) -> None:
        blocks = build_system(context)
        index = next(i for i, b in enumerate(blocks) if b.cacheable)
        cached = sum(len(b.text) for b in blocks[: index + 1])
        total = sum(len(b.text) for b in blocks)

        assert cached / total > 0.6, "the cacheable prefix is too small to be worth caching"


class TestRequests:
    def test_a_draft_request_carries_topic_and_angle(self, context: PromptContext) -> None:
        request = draft_request(context, topic="sumcheck", angle="it got simpler as it generalised")

        assert "sumcheck" in request.prompt
        assert "it got simpler as it generalised" in request.prompt

    def test_regeneration_feedback_is_specific(self, context: PromptContext) -> None:
        # Regenerating with an unchanged prompt just rolls the dice again.
        request = draft_request(
            context, topic="t", angle="a", feedback="the opening was a generic hook"
        )

        assert "the opening was a generic hook" in request.prompt
        assert "rejected" in request.prompt

    def test_an_idea_request_carries_the_type_allocation(self, context: PromptContext) -> None:
        request = idea_request(context, count=20, content_types={"technical": 8, "humor": 2})

        assert "technical: 8" in request.prompt
        assert request.json_schema is IDEA_SCHEMA


class TestMockProvider:
    def test_output_is_deterministic(self) -> None:
        provider = MockProvider()
        request = LLMRequest(system=system_prompt("persona"), prompt="write something")

        assert provider.complete(request).text == provider.complete(request).text

    def test_different_prompts_differ(self) -> None:
        provider = MockProvider()
        first = provider.complete(LLMRequest(system=(), prompt="one"))
        second = provider.complete(LLMRequest(system=(), prompt="two"))

        assert first.text != second.text

    def test_output_obeys_the_house_style(self) -> None:
        # Mock output feeds the evaluators in pipeline tests, so lorem ipsum
        # would let everything pass for the wrong reason.
        from contentsys.content.sanitize import EM_DASH

        provider = MockProvider()
        text = provider.complete(LLMRequest(system=(), prompt="x")).text

        assert EM_DASH not in text
        assert text == text.lower()

    def test_usage_separates_cached_from_fresh_tokens(self) -> None:
        provider = MockProvider()
        request = LLMRequest(
            system=system_prompt("a long stable persona block", "volatile", cache_through=0),
            prompt="write",
        )

        usage = provider.complete(request).usage

        assert usage.cache_read_tokens > 0
        assert usage.output_tokens > 0

    def test_structured_output_satisfies_the_schema(self) -> None:
        provider = MockProvider()
        request = LLMRequest(system=(), prompt="ideas", json_schema=IDEA_SCHEMA)

        result = provider.complete_json(request)

        assert "ideas" in result
        assert result["ideas"]
        first = result["ideas"][0]
        assert set(IDEA_SCHEMA["properties"]["ideas"]["items"]["required"]) <= set(first)
        assert isinstance(first["needs_experience"], bool)
        assert 0 <= first["novelty"] <= 10

    def test_complete_json_without_a_schema_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="json_schema"):
            MockProvider().complete_json(LLMRequest(system=(), prompt="x"))

    def test_failures_can_be_forced(self) -> None:
        from contentsys.llm.base import LLMError

        provider = MockProvider(fail_on="explode")

        with pytest.raises(LLMError):
            provider.complete(LLMRequest(system=(), prompt="please explode"))


class TestRegistry:
    def test_mock_is_built_by_name(self) -> None:
        assert build_provider(ProviderName.MOCK).name == "mock"

    def test_a_string_name_works(self) -> None:
        assert build_provider("mock").name == "mock"

    def test_an_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_provider("telepathy")

    def test_the_anthropic_provider_explains_a_missing_key(self) -> None:
        # The confusing part is that a Claude subscription looks like it should
        # work, so the error has to say why it does not.
        from contentsys.llm.anthropic_api import AnthropicProvider
        from contentsys.llm.base import LLMUnavailable

        with pytest.raises(LLMUnavailable, match="does not include API access"):
            _ = AnthropicProvider(api_key=None).client


class TestAgentSDKParsing:
    def test_json_is_extracted_from_a_markdown_fence(self) -> None:
        # Models fence JSON often enough that failing on it would be a
        # self-inflicted error rate rather than a real one.
        from contentsys.llm.agent_sdk import _parse_json

        assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_is_extracted_from_surrounding_prose(self) -> None:
        from contentsys.llm.agent_sdk import _parse_json

        assert _parse_json('Here you go: {"a": 1} hope that helps') == {"a": 1}

    def test_unparseable_output_raises_with_context(self) -> None:
        from contentsys.llm.agent_sdk import _parse_json
        from contentsys.llm.base import LLMError

        with pytest.raises(LLMError, match="expected JSON"):
            _parse_json("no json here at all")

    def test_an_auth_failure_is_unavailable_not_a_refusal(self) -> None:
        # The distinction decides whether retrying is worth anything.
        from contentsys.llm.agent_sdk import _translate_error
        from contentsys.llm.base import LLMRefusal, LLMUnavailable

        assert isinstance(_translate_error("authentication_failed"), LLMUnavailable)
        assert isinstance(_translate_error("refusal"), LLMRefusal)


def test_prompt_context_reports_verified_experience(session: Session, rules: ContentRules) -> None:
    context = PromptContext(platform=Platform.X, voice=analyse(REAL_POSTS), rules=rules)

    assert not context.has_verified_experience

    context.experiences = [Experience(summary="real thing", confidence=Confidence.STATED)]
    assert context.has_verified_experience

    context.experiences = [Experience(summary="unsure", confidence=Confidence.INFERRED)]
    assert not context.has_verified_experience
