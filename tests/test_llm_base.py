from __future__ import annotations

import pytest

from contentsys.llm import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    SystemBlock,
    Usage,
    collect_usage,
    system_prompt,
)


class TestSystemPrompt:
    def test_plain_strings_become_blocks(self) -> None:
        blocks = system_prompt("persona", "voice")

        assert [b.text for b in blocks] == ["persona", "voice"]
        assert not any(b.cacheable for b in blocks)

    def test_empty_fragments_are_dropped(self) -> None:
        # Callers pass optional sections without branching around them.
        blocks = system_prompt("persona", "", None or "", "voice")

        assert [b.text for b in blocks] == ["persona", "voice"]

    def test_cache_through_marks_the_stable_prefix(self) -> None:
        blocks = system_prompt("persona", "voice", "today's topic", cache_through=1)

        assert [b.cacheable for b in blocks] == [False, True, False]

    def test_cache_through_accepts_negative_indices(self) -> None:
        blocks = system_prompt("persona", "voice", "slop rules", cache_through=-1)

        assert blocks[-1].cacheable is True

    def test_cache_through_out_of_range_raises(self) -> None:
        with pytest.raises(IndexError):
            system_prompt("persona", cache_through=4)

    def test_cache_through_on_empty_prompt_is_a_noop(self) -> None:
        assert system_prompt("", cache_through=0) == ()

    def test_preserves_an_explicitly_marked_block(self) -> None:
        blocks = system_prompt(SystemBlock("persona", cacheable=True), "topic")

        assert blocks[0].cacheable is True


class TestLLMRequest:
    def test_system_text_joins_blocks(self) -> None:
        request = LLMRequest(system=system_prompt("persona", "voice"), prompt="write")

        assert request.system_text() == "persona\n\nvoice"

    def test_system_text_on_an_empty_prompt(self) -> None:
        request = LLMRequest(system=(), prompt="write")

        assert request.system_text() == ""

    def test_carries_no_temperature(self) -> None:
        # The target models reject sampling parameters, so the field must not
        # exist rather than be silently ignored at the boundary.
        assert not hasattr(LLMRequest(system=(), prompt="x"), "temperature")


class TestUsage:
    def test_addition_accumulates_every_field(self) -> None:
        total = Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40)

        assert total == Usage(11, 22, 33, 44)

    def test_collect_sums_a_batch(self) -> None:
        responses = [
            LLMResponse(text="a", model="m", usage=Usage(input_tokens=5, output_tokens=1)),
            LLMResponse(text="b", model="m", usage=Usage(input_tokens=7, cache_read_tokens=100)),
        ]

        total = collect_usage(responses)

        assert total.input_tokens == 12
        assert total.output_tokens == 1
        assert total.cache_read_tokens == 100

    def test_collect_on_an_empty_batch(self) -> None:
        assert collect_usage([]) == Usage()


class TestProtocol:
    def test_a_minimal_implementation_satisfies_the_protocol(self) -> None:
        class Stub:
            name = "stub"

            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(text="", model="stub")

            def complete_json(self, request: LLMRequest) -> dict:
                return {}

        assert isinstance(Stub(), LLMProvider)

    def test_a_class_missing_complete_does_not(self) -> None:
        class Incomplete:
            name = "incomplete"

        assert not isinstance(Incomplete(), LLMProvider)
