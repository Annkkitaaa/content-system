"""The diagram engine.

The renderer is deterministic, so all of this runs without a model call. That
is the point of the spec/render split: the half most likely to break silently
is the half that can be tested exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contentsys.config import Platform, Settings, get_settings
from contentsys.llm.base import LLMError, LLMRequest
from contentsys.prompts import PromptContext
from contentsys.visuals import (
    Column,
    DiagramKind,
    DiagramSpec,
    Node,
    SpecError,
    diagram_path,
    generate_diagram,
    render,
    wants_diagram,
)
from contentsys.visuals.render import _chars_that_fit, _fit
from contentsys.visuals.spec import MAX_NODES
from contentsys.voice.surface import analyse


def chain(**overrides) -> DiagramSpec:
    base = {
        "kind": DiagramKind.CHAIN,
        "title": "How Spartan verifies millions of constraints",
        "nodes": [
            Node("Computation", "the thing you ran"),
            Node("R1CS", "every operation as arithmetic constraints"),
            Node("Sum-check", "an exponential claim becomes one evaluation", highlight=True),
        ],
    }
    return DiagramSpec(**{**base, **overrides})


class TestValidation:
    def test_a_good_chain_validates(self) -> None:
        assert chain().validate() is not None

    def test_a_title_is_required(self) -> None:
        with pytest.raises(SpecError, match="title"):
            chain(title="   ").validate()

    def test_one_node_is_not_a_diagram(self) -> None:
        with pytest.raises(SpecError, match="at least two nodes"):
            chain(nodes=[Node("Alone")]).validate()

    def test_too_many_nodes_are_rejected(self) -> None:
        # The constraint is a phone in a scrolling timeline, not taste.
        with pytest.raises(SpecError, match="legible on a phone"):
            chain(nodes=[Node(f"Step {n}") for n in range(MAX_NODES + 1)]).validate()

    def test_highlighting_everything_is_rejected(self) -> None:
        # Highlighting several steps highlights none of them.
        with pytest.raises(SpecError, match="at most one node"):
            chain(nodes=[Node("A", highlight=True), Node("B", highlight=True)]).validate()

    def test_a_comparison_needs_two_columns(self) -> None:
        with pytest.raises(SpecError, match="at least two columns"):
            DiagramSpec(
                kind=DiagramKind.COMPARISON,
                title="t",
                columns=[Column("Only one", ["a"])],
            ).validate()

    def test_a_comparison_column_needs_rows(self) -> None:
        with pytest.raises(SpecError, match="at least one row"):
            DiagramSpec(
                kind=DiagramKind.COMPARISON,
                title="t",
                columns=[Column("A", ["x"]), Column("B", [])],
            ).validate()


class TestTruncation:
    def test_a_long_label_is_clipped_rather_than_rejected(self) -> None:
        # A model that writes a slightly long label has still had the right
        # idea. Losing the diagram over a few characters is the worse outcome.
        spec = chain(nodes=[Node("x" * 200), Node("B")]).truncated()

        assert len(spec.nodes[0].label) < 60
        spec.validate()

    def test_clipping_prefers_a_word_boundary(self) -> None:
        spec = chain(
            nodes=[Node("the quick brown fox jumps over the lazy dog and keeps going"), Node("B")]
        ).truncated()

        assert not spec.nodes[0].label.startswith("the quick brown fox jumps over the lazy dogg")

    def test_highlight_survives_truncation(self) -> None:
        assert chain().truncated().nodes[2].highlight


class TestAltText:
    def test_supplied_alt_text_is_used(self) -> None:
        assert chain(alt_text="a custom description").describe() == "a custom description"

    def test_alt_text_is_generated_when_missing(self) -> None:
        # An image nobody can read is worse than no image, and both platforms
        # support a description, so this never returns empty.
        described = chain().describe()

        assert "Spartan" in described
        assert "R1CS" in described

    def test_a_comparison_describes_both_sides(self) -> None:
        spec = DiagramSpec(
            kind=DiagramKind.COMPARISON,
            title="Two systems",
            columns=[Column("Groth16", ["a"]), Column("Spartan", ["b"])],
        )

        assert "Groth16 against Spartan" in spec.describe()


class TestSerialisation:
    def test_a_spec_survives_a_round_trip(self) -> None:
        # Storing the spec beside the image is what lets a style change
        # regenerate every past diagram without a model call.
        original = chain(caption="a caption")

        restored = DiagramSpec.from_dict(original.to_dict())

        assert restored.kind is original.kind
        assert restored.title == original.title
        assert [n.label for n in restored.nodes] == [n.label for n in original.nodes]
        assert restored.nodes[2].highlight

    def test_an_unknown_kind_is_rejected_rather_than_guessed(self) -> None:
        # Guessing which kind was meant produces a confidently wrong diagram.
        with pytest.raises(SpecError, match="unknown diagram kind"):
            DiagramSpec.from_dict({"kind": "sankey", "title": "t"})

    def test_malformed_nodes_are_skipped(self) -> None:
        spec = DiagramSpec.from_dict(
            {"kind": "chain", "title": "t", "nodes": [{"label": "A"}, "not a dict", {"label": "B"}]}
        )

        assert [n.label for n in spec.nodes] == ["A", "B"]


class TestTextFitting:
    def test_wrapping_respects_the_width(self) -> None:
        wrapped = _fit("one two three four five six seven eight", 12)

        assert all(len(line) <= 12 for line in wrapped.splitlines())

    def test_an_unbreakable_word_is_broken(self) -> None:
        # A single word wider than the box runs past the edge no matter how
        # the rest is wrapped, so it has to be split.
        wrapped = _fit("supercalifragilisticexpialidocious", 10)

        assert all(len(line) <= 10 for line in wrapped.splitlines())

    def test_line_count_is_capped(self) -> None:
        wrapped = _fit(" ".join(["word"] * 40), 10, max_lines=2)

        assert len(wrapped.splitlines()) == 2

    def test_narrower_boxes_fit_fewer_characters(self) -> None:
        # The first version of this used a magic multiplier that happened to
        # work at one box count and overflowed at others.
        assert _chars_that_fit(30, 12.5) > _chars_that_fit(15, 12.5)

    def test_larger_type_fits_fewer_characters(self) -> None:
        assert _chars_that_fit(20, 12.5) > _chars_that_fit(20, 26.0)


class TestRendering:
    def test_a_chain_renders_to_a_file(self, tmp_path: Path) -> None:
        path = render(chain(), tmp_path / "chain.png")

        assert path.exists()
        assert path.stat().st_size > 5000

    def test_a_comparison_renders(self, tmp_path: Path) -> None:
        spec = DiagramSpec(
            kind=DiagramKind.COMPARISON,
            title="Groth16 and Spartan start in the same place",
            columns=[
                Column("Groth16", ["R1CS", "QAP", "Trusted setup"]),
                Column("Spartan", ["R1CS", "Multilinear extensions", "Transparent"]),
            ],
        )

        assert render(spec, tmp_path / "cmp.png").stat().st_size > 5000

    @pytest.mark.parametrize("kind", [DiagramKind.CHAIN, DiagramKind.FLOW, DiagramKind.TIMELINE])
    def test_every_sequence_kind_renders(self, kind: DiagramKind, tmp_path: Path) -> None:
        assert render(chain(kind=kind), tmp_path / f"{kind.value}.png").exists()

    def test_rendering_is_deterministic(self, tmp_path: Path) -> None:
        # The whole reason the model does not draw: the same spec must always
        # produce the same image, or a style change cannot be verified.
        first = render(chain(), tmp_path / "a.png").read_bytes()
        second = render(chain(), tmp_path / "b.png").read_bytes()

        assert first == second

    def test_missing_directories_are_created(self, tmp_path: Path) -> None:
        path = render(chain(), tmp_path / "deep" / "nested" / "d.png")

        assert path.exists()

    def test_an_overlong_label_still_renders(self, tmp_path: Path) -> None:
        # Truncation runs before validation inside render, so a long label is
        # an ellipsis rather than a layout failure nobody notices.
        spec = chain(nodes=[Node("x" * 300, "y" * 300), Node("B")])

        assert render(spec, tmp_path / "long.png").exists()

    def test_an_invalid_spec_fails_before_writing_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "never.png"

        with pytest.raises(SpecError):
            render(chain(nodes=[Node("Alone")]), path)

        assert not path.exists()


class TestPolicy:
    @pytest.fixture
    def context(self) -> PromptContext:
        get_settings.cache_clear()
        return PromptContext(
            platform=Platform.X, voice=analyse(["i think this"]), rules=Settings().content_rules
        )

    def test_every_linkedin_post_is_eligible(self, context: PromptContext) -> None:
        context.platform = Platform.LINKEDIN

        assert wants_diagram(context, "thoughtful_opinion")

    def test_structural_x_posts_are_eligible(self, context: PromptContext) -> None:
        assert wants_diagram(context, "technical")
        assert wants_diagram(context, "mini_analysis")

    def test_conversational_x_posts_are_not(self, context: PromptContext) -> None:
        # An image on a one line thought is decoration, and decoration costs a
        # generation call and a review decision.
        assert not wants_diagram(context, "humor")
        assert not wants_diagram(context, "personal_reflection")

    def test_paths_are_grouped_by_platform(self, tmp_path: Path) -> None:
        path = diagram_path(tmp_path, Platform.X, "spartan chain/1")

        assert path.parent.name == "x"
        assert "/" not in path.name


class TestGeneration:
    @pytest.fixture
    def context(self) -> PromptContext:
        get_settings.cache_clear()
        return PromptContext(
            platform=Platform.X, voice=analyse(["i think this"]), rules=Settings().content_rules
        )

    def test_a_valid_response_becomes_a_spec(self, context: PromptContext) -> None:
        class Good:
            name = "good"

            def complete(self, request: LLMRequest):
                raise NotImplementedError

            def complete_json(self, request: LLMRequest) -> dict:
                return {
                    "kind": "chain",
                    "title": "A real title",
                    "alt_text": "described",
                    "nodes": [{"label": "A"}, {"label": "B", "highlight": True}],
                }

        spec = generate_diagram(Good(), context, content="a post")

        assert spec is not None
        assert spec.kind is DiagramKind.CHAIN

    def test_a_malformed_response_yields_none_not_an_exception(
        self, context: PromptContext
    ) -> None:
        # A missing diagram is a small loss. A failed weekly run over one bad
        # JSON response is not.
        class Broken:
            name = "broken"

            def complete(self, request: LLMRequest):
                raise NotImplementedError

            def complete_json(self, request: LLMRequest) -> dict:
                return {"kind": "sankey", "title": "nope"}

        assert generate_diagram(Broken(), context, content="a post") is None

    def test_a_provider_failure_yields_none(self, context: PromptContext) -> None:
        class Failing:
            name = "failing"

            def complete(self, request: LLMRequest):
                raise NotImplementedError

            def complete_json(self, request: LLMRequest) -> dict:
                raise LLMError("upstream is down")

        assert generate_diagram(Failing(), context, content="a post") is None

    def test_a_structureless_response_yields_none(self, context: PromptContext) -> None:
        class Thin:
            name = "thin"

            def complete(self, request: LLMRequest):
                raise NotImplementedError

            def complete_json(self, request: LLMRequest) -> dict:
                return {"kind": "chain", "title": "t", "alt_text": "a", "nodes": [{"label": "A"}]}

        assert generate_diagram(Thin(), context, content="a one line opinion") is None
