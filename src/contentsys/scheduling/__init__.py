"""Weekly scheduling.

Slots carry deliberate jitter: evenly spaced posts on round numbered minutes
are a behavioural signal the ranking model uses against an account.
"""

from __future__ import annotations

from contentsys.scheduling.slots import (
    WEEKDAYS,
    Slot,
    daily_slots,
    describe,
    week_starting,
    weekly_slots,
)

__all__ = [
    "WEEKDAYS",
    "Slot",
    "daily_slots",
    "describe",
    "week_starting",
    "weekly_slots",
]
