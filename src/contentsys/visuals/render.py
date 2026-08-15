"""Turning a spec into an image.

Deterministic: the same spec always produces the same file. That is what makes
this testable and what lets a style change regenerate every past diagram
without a model call.

Two constraints drove the design, and both come from where these are actually
seen rather than from taste.

They are looked at on a phone, in a scrolling timeline, at maybe a third of
their rendered size. So: few elements, large type, high contrast, and a hard
cap on how much text a node may carry. A diagram that needs to be opened to be
read has already failed.

They accumulate on a profile. So the style is fixed and slightly austere
rather than expressive. Twenty diagrams that look like one person made them
say something; twenty that each look different say nothing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Headless. Must be set before pyplot is imported, or a machine without a
# display fails at import time rather than at draw time.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from contentsys.visuals.spec import DiagramKind, DiagramSpec


class Theme:
    """The house style. One place, so every diagram matches.

    The palette is deliberately narrow: paper, ink, one accent. Colour carries
    meaning here (the accent marks the step where the interesting thing
    happens) so spending it on decoration would make it stop working.
    """

    paper = "#FBFAF7"
    ink = "#16161A"
    muted = "#6B6B76"
    line = "#D6D3CC"
    box = "#FFFFFF"
    accent = "#C2410C"
    accent_soft = "#FDE8DC"

    title_size = 26
    label_size = 17
    note_size = 12.5
    caption_size = 13

    #: matplotlib only accepts a fixed set of weight names, and an unknown one
    #: falls back silently after printing a warning per draw call.
    heading_weight = "bold"
    label_weight = "normal"

    #: 16:9 at 100 dpi. Both platforms crop toward this, and it is the shape
    #: that survives a timeline without letterboxing.
    figsize = (16.0, 9.0)
    dpi = 100

    #: The axes is pinned to the whole figure so that one axis unit is a fixed
    #: number of pixels. Text fitting depends on that being predictable, and
    #: with default subplot margins it is not.
    px_per_unit = figsize[0] * dpi / 100.0

    #: Mean glyph width as a fraction of point size, for the default sans
    #: face. Measured rather than guessed, and deliberately on the generous
    #: side: a label that wraps one line early is fine, one that overflows its
    #: box is not.
    glyph_ratio = 0.60


def _chars_that_fit(width_units: float, font_size: float, *, padding_units: float = 2.6) -> int:
    """How many characters fit across a box of this width.

    Computed from real geometry rather than a magic multiplier. The first
    version of this used ``int(width * 1.15)``, which happened to be roughly
    right at one box count and overflowed at others, and the failure is not
    visible until you look at the rendered file.
    """
    usable_px = max(0.0, width_units - padding_units) * Theme.px_per_unit
    glyph_px = Theme.glyph_ratio * font_size * Theme.dpi / 72.0
    return max(8, int(usable_px / glyph_px))


def _fit(text: str, width_chars: int, *, max_lines: int = 3) -> str:
    """Wrap a label, breaking over-long single words rather than overflowing.

    Done here instead of by matplotlib because the box is sized from the
    result, so the wrapping has to be known before the box is drawn.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        # A single word wider than the box has to be broken, or it runs past
        # the edge no matter how the rest is wrapped.
        while len(word) > width_chars:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[: width_chars - 1] + chr(0x2010))
            word = word[width_chars - 1 :]
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(1, width_chars - 1)].rstrip() + chr(0x2026)
    return "\n".join(lines)


def _canvas(spec: DiagramSpec) -> tuple[plt.Figure, plt.Axes]:
    figure = plt.figure(figsize=Theme.figsize, dpi=Theme.dpi)
    # Pin the axes to the entire figure so one unit is exactly px_per_unit
    # pixels. Everything in _chars_that_fit depends on that.
    axes = figure.add_axes((0, 0, 1, 1))
    figure.patch.set_facecolor(Theme.paper)
    axes.set_facecolor(Theme.paper)
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")

    title = _fit(spec.title, _chars_that_fit(88, Theme.title_size, padding_units=0), max_lines=2)
    axes.text(
        6,
        90,
        title,
        fontsize=Theme.title_size,
        color=Theme.ink,
        ha="left",
        va="top",
        weight=Theme.heading_weight,
        linespacing=1.25,
    )
    # A rule under the title, cheap, and it makes the whole thing look composed.
    rule_y = 80 if title.count("\n") else 84
    axes.plot([6, 94], [rule_y, rule_y], color=Theme.line, linewidth=1.4, solid_capstyle="round")

    if spec.caption:
        axes.text(
            6,
            7,
            _fit(
                spec.caption, _chars_that_fit(88, Theme.caption_size, padding_units=0), max_lines=2
            ),
            fontsize=Theme.caption_size,
            color=Theme.muted,
            ha="left",
            va="bottom",
        )
    return figure, axes


def _box(
    axes: plt.Axes, x: float, y: float, width: float, height: float, *, highlight: bool
) -> None:
    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=2.0 if highlight else 1.3,
            edgecolor=Theme.accent if highlight else Theme.line,
            facecolor=Theme.accent_soft if highlight else Theme.box,
            zorder=2,
        )
    )


def _arrow(axes: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            color=Theme.muted,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def _draw_sequence(axes: plt.Axes, spec: DiagramSpec) -> None:
    """Chain, flow and timeline all draw as a left to right sequence.

    They differ in what they mean, not in what they look like, and inventing
    three visual languages for one shape would make the set less coherent for
    no gain in clarity.
    """
    count = len(spec.nodes)
    margin, gap = 6.0, 3.2
    available = 88.0 - gap * (count - 1)
    width = available / count
    has_notes = any(node.note for node in spec.nodes)

    # Sit the row a little above centre. Optical centre is higher than
    # geometric centre, and the caption occupies the bottom.
    height = 34.0 if has_notes else 24.0
    y = 34.0 if has_notes else 40.0

    label_chars = _chars_that_fit(width, Theme.label_size)
    note_chars = _chars_that_fit(width, Theme.note_size)

    for index, node in enumerate(spec.nodes):
        x = margin + index * (width + gap)
        _box(axes, x, y, width, height, highlight=node.highlight)

        centre = x + width / 2
        label = _fit(node.label, label_chars, max_lines=2)
        if node.note:
            note = _fit(node.note, note_chars, max_lines=3)
            # Split the box between label and note rather than using fixed
            # offsets, so a two line label does not collide with the note.
            axes.text(
                centre,
                y + height * 0.68,
                label,
                fontsize=Theme.label_size,
                color=Theme.ink,
                ha="center",
                va="center",
                weight=Theme.label_weight,
                linespacing=1.2,
                zorder=3,
            )
            axes.text(
                centre,
                y + height * 0.28,
                note,
                fontsize=Theme.note_size,
                color=Theme.muted,
                ha="center",
                va="center",
                linespacing=1.35,
                zorder=3,
            )
        else:
            axes.text(
                centre,
                y + height / 2,
                label,
                fontsize=Theme.label_size,
                color=Theme.ink,
                ha="center",
                va="center",
                weight=Theme.label_weight,
                linespacing=1.2,
                zorder=3,
            )

        if index < count - 1:
            _arrow(axes, (x + width + 0.4, y + height / 2), (x + width + gap - 0.4, y + height / 2))

    if spec.kind is DiagramKind.TIMELINE:
        for index in range(count):
            x = margin + index * (width + gap) + width / 2
            axes.text(
                x,
                y - 4.5,
                str(index + 1),
                fontsize=Theme.note_size,
                color=Theme.muted,
                ha="center",
                va="top",
            )


def _draw_comparison(axes: plt.Axes, spec: DiagramSpec) -> None:
    columns = spec.columns
    margin, gap = 6.0, 4.0
    width = (88.0 - gap * (len(columns) - 1)) / len(columns)
    top, bottom = 68.0, 16.0

    rows = max(len(column.rows) for column in columns)
    row_height = (top - bottom) / max(rows, 1)
    row_size = Theme.note_size + 1.5
    # Rows are left aligned inside the box, so they lose the left inset as
    # well as the usual padding.
    row_chars = _chars_that_fit(width, row_size, padding_units=4.4)
    title_chars = _chars_that_fit(width, Theme.label_size + 1, padding_units=0)

    for index, column in enumerate(columns):
        x = margin + index * (width + gap)
        axes.text(
            x + width / 2,
            top + 2.5,
            _fit(column.title, title_chars, max_lines=1),
            fontsize=Theme.label_size + 1,
            color=Theme.ink,
            ha="center",
            va="bottom",
            weight=Theme.heading_weight,
        )
        _box(axes, x, bottom, width, top - bottom, highlight=False)

        for row_index, row in enumerate(column.rows):
            y = top - (row_index + 0.5) * row_height
            axes.text(
                x + 2.4,
                y,
                _fit(row, row_chars, max_lines=2),
                fontsize=row_size,
                color=Theme.ink,
                ha="left",
                va="center",
                linespacing=1.25,
                zorder=3,
            )
            if row_index:
                axes.plot(
                    [x + 1.5, x + width - 1.5],
                    [y + row_height / 2, y + row_height / 2],
                    color=Theme.line,
                    linewidth=0.9,
                    zorder=2,
                )


def render(spec: DiagramSpec, path: Path) -> Path:
    """Draw a spec to a PNG.

    The spec is validated and clipped first, so an over-long label becomes an
    ellipsis rather than a layout failure nobody notices until it is posted.
    """
    prepared = spec.truncated()
    prepared.validate()

    figure, axes = _canvas(prepared)
    if prepared.kind is DiagramKind.COMPARISON:
        _draw_comparison(axes, prepared)
    else:
        _draw_sequence(axes, prepared)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=Theme.paper, bbox_inches="tight", pad_inches=0.35)
    plt.close(figure)
    return path
