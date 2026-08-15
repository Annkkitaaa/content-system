"""Orchestration.

The only place stages are sequenced. Everything it calls is independently
usable and independently tested, so this package stays a sequence rather than
a place where logic hides.
"""

from __future__ import annotations

from contentsys.pipeline.weekly import WeeklyResult, record_snapshot, run_weekly

__all__ = ["WeeklyResult", "record_snapshot", "run_weekly"]
