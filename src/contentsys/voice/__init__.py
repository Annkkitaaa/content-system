"""The voice engine.

Two layers. :mod:`contentsys.voice.surface` measures mechanics with no model
call, so it is exact and testable. The semantic layer, which models how the
owner actually thinks and argues, arrives with the generation phase.

The second one drives generation. The first one checks it. Getting that the
wrong way round produces text that counts lowercase letters correctly and
still sounds like nobody.
"""

from __future__ import annotations

from contentsys.voice.diff import Change, EditAnalysis, analyse_edit, word_diff
from contentsys.voice.memory import (
    ACTIVE_CONFIDENCE,
    LearningReport,
    active_preferences,
    forget,
    learn_from,
    record_edit,
)
from contentsys.voice.profile import (
    MIN_USEFUL_SAMPLES,
    active_profile,
    build_profile,
    load_surface,
    samples_for,
)
from contentsys.voice.surface import SurfaceProfile, analyse, compare

__all__ = [
    "ACTIVE_CONFIDENCE",
    "MIN_USEFUL_SAMPLES",
    "Change",
    "EditAnalysis",
    "LearningReport",
    "SurfaceProfile",
    "active_preferences",
    "active_profile",
    "analyse",
    "analyse_edit",
    "build_profile",
    "compare",
    "forget",
    "learn_from",
    "load_surface",
    "record_edit",
    "samples_for",
    "word_diff",
]
