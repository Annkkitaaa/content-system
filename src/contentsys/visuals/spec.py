"""What a diagram says, separately from how it looks.

A spec is data: nodes, edges, a kind. It carries no colours, no fonts, no
coordinates. The renderer owns all of that.

That split is the whole design. A model is good at deciding that a post about
Spartan is a four step reduction chain and terrible at deciding where the
boxes go. Code is the reverse. Keeping them apart means rendering is
deterministic and testable, every diagram looks like it came from the same
person, and a style change can regenerate every past image without asking a
model to reinvent it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DiagramKind(StrEnum):
    """The shapes this account's content actually takes.

    Deliberately few. A long list of chart types invites a model to pick an
    exotic one when a chain would have been clearer.
    """

    #: A sequence of reductions, each step transforming the problem into
    #: something easier. The signature shape of this account's writing.
    CHAIN = "chain"
    #: A protocol with an exchange between two parties, or rounds.
    FLOW = "flow"
    #: Two systems set side by side on the dimensions that actually differ.
    COMPARISON = "comparison"
    #: An ordered sequence: rounds of a protocol, or an incident unfolding.
    TIMELINE = "timeline"


#: A node label longer than this stops being legible at the size these images
#: are actually viewed at, which is a phone in a scrolling timeline.
MAX_LABEL = 42
MAX_CAPTION = 90
MAX_NODES = 7
MAX_COLUMNS = 3


class SpecError(ValueError):
    """A spec that cannot be rendered into something worth posting."""


@dataclass
class Node:
    """One box.

    ``label`` is the thing itself. ``note`` is what it costs or buys, which is
    the part that carries the argument in a reduction chain.
    """

    label: str
    note: str | None = None
    #: Marks the step where the interesting thing happens. Exactly one node
    #: may be highlighted; more than one and nothing is.
    highlight: bool = False


@dataclass
class Column:
    """One side of a comparison."""

    title: str
    rows: list[str] = field(default_factory=list)


@dataclass
class DiagramSpec:
    """A complete diagram, before anything is drawn."""

    kind: DiagramKind
    title: str
    nodes: list[Node] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    caption: str | None = None

    #: Reader-facing description. Required rather than optional: an image
    #: nobody can read is worse than no image, and both platforms support it.
    alt_text: str | None = None

    def validate(self) -> DiagramSpec:
        """Reject specs that would render into something unreadable.

        Checked here rather than at draw time so a bad spec fails before a
        file is written, and so the failure names the field rather than
        surfacing as a layout artefact nobody notices.
        """
        if not self.title.strip():
            raise SpecError("a diagram needs a title")

        if self.kind is DiagramKind.COMPARISON:
            if len(self.columns) < 2:
                raise SpecError("a comparison needs at least two columns")
            if len(self.columns) > MAX_COLUMNS:
                raise SpecError(
                    f"{len(self.columns)} columns will not be legible on a phone, "
                    f"the limit is {MAX_COLUMNS}"
                )
            if any(not column.title.strip() for column in self.columns):
                raise SpecError("every comparison column needs a title")
            if any(not column.rows for column in self.columns):
                raise SpecError("every comparison column needs at least one row")
        else:
            if len(self.nodes) < 2:
                raise SpecError(f"a {self.kind.value} needs at least two nodes")
            if len(self.nodes) > MAX_NODES:
                raise SpecError(
                    f"{len(self.nodes)} nodes will not be legible on a phone, "
                    f"the limit is {MAX_NODES}"
                )
            if any(not node.label.strip() for node in self.nodes):
                raise SpecError("every node needs a label")
            if sum(1 for node in self.nodes if node.highlight) > 1:
                # Highlighting everything highlights nothing.
                raise SpecError("at most one node may be highlighted")

        return self

    def truncated(self) -> DiagramSpec:
        """Return a copy with over-long text shortened.

        Preferred over rejecting: a model that writes a slightly long label
        has still had the right idea, and losing the diagram over a few
        characters is a worse outcome than an ellipsis.
        """
        return DiagramSpec(
            kind=self.kind,
            title=_clip(self.title, MAX_LABEL + 20),
            nodes=[
                Node(
                    label=_clip(node.label, MAX_LABEL),
                    note=_clip(node.note, MAX_LABEL) if node.note else None,
                    highlight=node.highlight,
                )
                for node in self.nodes
            ],
            columns=[
                Column(
                    title=_clip(column.title, MAX_LABEL),
                    rows=[_clip(r, MAX_LABEL) for r in column.rows],
                )
                for column in self.columns
            ],
            caption=_clip(self.caption, MAX_CAPTION) if self.caption else None,
            alt_text=self.alt_text,
        )

    def describe(self) -> str:
        """Alt text, generated when none was supplied.

        A diagram with no description is unreadable to anyone using a screen
        reader, so this always produces something rather than leaving it
        empty.
        """
        if self.alt_text:
            return self.alt_text
        if self.kind is DiagramKind.COMPARISON:
            sides = " against ".join(column.title for column in self.columns)
            return f"{self.title}. A comparison of {sides}."
        steps = ", then ".join(node.label for node in self.nodes)
        shape = {
            DiagramKind.CHAIN: "A reduction chain",
            DiagramKind.FLOW: "A flow",
            DiagramKind.TIMELINE: "A sequence",
        }[self.kind]
        return f"{self.title}. {shape}: {steps}."

    def to_dict(self) -> dict[str, Any]:
        """Serialised form, stored next to the rendered file.

        Keeping the spec means a style change regenerates every past image
        without a model call.
        """
        return {
            "kind": self.kind.value,
            "title": self.title,
            "caption": self.caption,
            "alt_text": self.describe(),
            "nodes": [
                {"label": n.label, "note": n.note, "highlight": n.highlight} for n in self.nodes
            ],
            "columns": [{"title": c.title, "rows": list(c.rows)} for c in self.columns],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DiagramSpec:
        """Rebuild from stored or model-produced JSON.

        Tolerant about missing optional fields, strict about the kind: an
        unrecognised kind means the model invented one, and guessing which it
        meant would produce a confidently wrong diagram.
        """
        raw_kind = str(payload.get("kind", "")).strip().lower()
        try:
            kind = DiagramKind(raw_kind)
        except ValueError as exc:
            valid = ", ".join(k.value for k in DiagramKind)
            raise SpecError(f"unknown diagram kind {raw_kind!r}, expected one of: {valid}") from exc

        return cls(
            kind=kind,
            title=str(payload.get("title", "")).strip(),
            caption=(str(payload["caption"]).strip() if payload.get("caption") else None),
            alt_text=(str(payload["alt_text"]).strip() if payload.get("alt_text") else None),
            nodes=[
                Node(
                    label=str(node.get("label", "")).strip(),
                    note=(str(node["note"]).strip() if node.get("note") else None),
                    highlight=bool(node.get("highlight", False)),
                )
                for node in payload.get("nodes", [])
                if isinstance(node, dict)
            ],
            columns=[
                Column(
                    title=str(column.get("title", "")).strip(),
                    rows=[str(row).strip() for row in column.get("rows", []) if str(row).strip()],
                )
                for column in payload.get("columns", [])
                if isinstance(column, dict)
            ],
        )


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    # Cut on a word boundary where one is close enough, so the result reads
    # as a shortened phrase rather than a severed one.
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(",;: ") + chr(0x2026)


#: The schema a model fills in. Kept beside the spec so the two cannot drift.
DIAGRAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": [k.value for k in DiagramKind]},
        "title": {"type": "string"},
        "caption": {"type": "string"},
        "alt_text": {"type": "string"},
        "nodes": {
            "type": "array",
            "maxItems": MAX_NODES,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "note": {"type": "string"},
                    "highlight": {"type": "boolean"},
                },
                "required": ["label"],
                "additionalProperties": False,
            },
        },
        "columns": {
            "type": "array",
            "maxItems": MAX_COLUMNS,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rows": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "rows"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["kind", "title", "alt_text"],
    "additionalProperties": False,
}
