"""Diagrams.

A model decides what a diagram says; deterministic code decides how it looks.
That split makes rendering testable, keeps every diagram looking like it came
from the same person, and lets a style change regenerate every past image
without a model call.
"""

from __future__ import annotations

from contentsys.visuals.generate import (
    diagram_path,
    diagram_request,
    generate_diagram,
    wants_diagram,
)
from contentsys.visuals.render import Theme, render
from contentsys.visuals.spec import (
    DIAGRAM_SCHEMA,
    Column,
    DiagramKind,
    DiagramSpec,
    Node,
    SpecError,
)

__all__ = [
    "DIAGRAM_SCHEMA",
    "Column",
    "DiagramKind",
    "DiagramSpec",
    "Node",
    "SpecError",
    "Theme",
    "diagram_path",
    "diagram_request",
    "generate_diagram",
    "render",
    "wants_diagram",
]
