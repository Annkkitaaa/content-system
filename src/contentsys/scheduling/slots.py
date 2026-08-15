"""Assigning dates and times to a week of posts.

Two things make this more than arithmetic.

**Timing must not look machine generated.** Evenly spaced posts on round
numbered minutes are a behavioural signal the ranking model uses against an
account. So every slot carries a random offset, and the offsets are drawn from
a seeded generator so a run is still reproducible: the same week regenerated
gives the same calendar, but the calendar does not look computed.

**A minimum gap has to actually hold.** Jitter can push two slots together, so
the gap is enforced after jittering rather than before, and a slot that cannot
be placed without violating it is nudged rather than dropped. Ten posts a day
inside five windows leaves real room, but the constraint has to survive the
randomness or it is decoration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from contentsys.config import Platform, PlatformSchedule, PostingWindow, Settings

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class Slot:
    """One scheduled posting time."""

    when: datetime
    platform: Platform
    window: str

    @property
    def day(self) -> str:
        return WEEKDAYS[self.when.weekday()]

    @property
    def date_text(self) -> str:
        return self.when.date().isoformat()

    @property
    def time_text(self) -> str:
        return self.when.strftime("%I:%M %p").lstrip("0")

    @property
    def timestamp(self) -> str:
        """ISO 8601 with offset, which is what a calendar import wants."""
        return self.when.isoformat()


def week_starting(reference: date | None = None, *, weekday: int = 0) -> date:
    """The next occurrence of ``weekday``, or today if it already matches.

    Defaults to Monday. Planning happens on a Sunday in practice, so the
    useful answer is almost always tomorrow rather than eight days out.
    """
    today = reference or date.today()
    ahead = (weekday - today.weekday()) % 7
    return today + timedelta(days=ahead)


def _window_for(index: int, windows: list[PostingWindow], count: int) -> PostingWindow:
    """Pick a window for the nth post of the day, spread by weight.

    Cumulative weights rather than round robin, so a window weighted 3 really
    does receive roughly three times as many posts as one weighted 1.
    """
    if not windows:
        raise ValueError("a platform schedule needs at least one posting window")

    total = sum(window.weight for window in windows)
    position = (index + 0.5) / count * total
    running = 0.0
    for window in windows:
        running += window.weight
        if position <= running:
            return window
    return windows[-1]


def _place_in_window(
    day: date,
    window: PostingWindow,
    slot_index: int,
    slots_in_window: int,
    jitter_minutes: int,
    rng: random.Random,
    tz: ZoneInfo,
) -> datetime:
    """A time inside a window, evenly divided then jittered."""
    span = window.duration_minutes
    # Divide the window into equal shares and aim at the middle of one, so two
    # posts in the same window do not start from the same point.
    share = span / max(1, slots_in_window)
    base = window.start_minutes + share * (slot_index + 0.5)

    if jitter_minutes:
        base += rng.uniform(-jitter_minutes, jitter_minutes)

    minute = round(base)
    minute = max(window.start_minutes, min(window.end_minutes - 1, minute))
    return datetime(day.year, day.month, day.day, minute // 60, minute % 60, tzinfo=tz)


def _enforce_gap(slots: list[datetime], gap_minutes: int) -> list[datetime]:
    """Push slots apart until the minimum gap holds.

    Applied after jittering, because jitter is what creates collisions. A slot
    is nudged later rather than dropped: a missing post is a hole in the week,
    and a post two minutes later is not a problem.
    """
    if gap_minutes <= 0 or not slots:
        return slots
    ordered = sorted(slots)
    gap = timedelta(minutes=gap_minutes)
    for index in range(1, len(ordered)):
        if ordered[index] - ordered[index - 1] < gap:
            ordered[index] = ordered[index - 1] + gap
    return ordered


def daily_slots(
    day: date,
    schedule: PlatformSchedule,
    platform: Platform,
    tz: ZoneInfo,
    rng: random.Random,
    *,
    count: int,
) -> list[Slot]:
    """Times for one day on one platform."""
    if count <= 0:
        return []

    windows = schedule.windows
    assignments: list[tuple[PostingWindow, int, int]] = []
    per_window: dict[str, int] = {}
    chosen: list[PostingWindow] = []

    for index in range(count):
        window = _window_for(index, windows, count)
        chosen.append(window)
        per_window[window.name] = per_window.get(window.name, 0) + 1

    seen: dict[str, int] = {}
    for window in chosen:
        position = seen.get(window.name, 0)
        seen[window.name] = position + 1
        assignments.append((window, position, per_window[window.name]))

    times = [
        _place_in_window(day, window, position, total, schedule.jitter_minutes, rng, tz)
        for window, position, total in assignments
    ]
    times = _enforce_gap(times, schedule.min_gap_minutes)

    names = {}
    for window, _, _ in assignments:
        names.setdefault(window.name, window)

    result: list[Slot] = []
    for when in times:
        minute = when.hour * 60 + when.minute
        window_name = next(
            (
                window.name
                for window, _, _ in assignments
                if window.start_minutes <= minute < window.end_minutes
            ),
            assignments[0][0].name,
        )
        result.append(Slot(when=when, platform=platform, window=window_name))
    return result


def weekly_slots(
    settings: Settings,
    *,
    start: date | None = None,
    x_posts: int | None = None,
    linkedin_posts: int | None = None,
    seed: int | None = None,
) -> dict[Platform, list[Slot]]:
    """A whole week of posting times.

    ``seed`` makes a run reproducible without making it look computed. Given
    the same seed the calendar is identical, which is what lets the export be
    regenerated and diffed.
    """
    tz = settings.tzinfo
    monday = start or week_starting()
    rng = random.Random(seed if seed is not None else monday.toordinal())

    x_schedule = settings.schedule.x
    per_day = x_posts if x_posts is not None else (x_schedule.posts_per_day or 10)

    x_slots: list[Slot] = []
    for offset in range(7):
        day = monday + timedelta(days=offset)
        x_slots.extend(daily_slots(day, x_schedule, Platform.X, tz, rng, count=per_day))

    linkedin_schedule = settings.schedule.linkedin
    total_linkedin = (
        linkedin_posts if linkedin_posts is not None else (linkedin_schedule.posts_per_week or 2)
    )
    linkedin_slots = _weekly_platform_slots(
        monday, linkedin_schedule, Platform.LINKEDIN, tz, rng, total_linkedin
    )

    return {Platform.X: x_slots, Platform.LINKEDIN: linkedin_slots}


def _weekly_platform_slots(
    monday: date,
    schedule: PlatformSchedule,
    platform: Platform,
    tz: ZoneInfo,
    rng: random.Random,
    total: int,
) -> list[Slot]:
    """Slots for a platform that posts a few times a week rather than daily."""
    if total <= 0:
        return []

    preferred = [day for day in schedule.preferred_days if day in WEEKDAYS]
    if not preferred:
        # Spread evenly across the week when no preference is configured.
        step = max(1, 7 // total)
        preferred = [WEEKDAYS[(index * step) % 7] for index in range(total)]

    slots: list[Slot] = []
    for index in range(total):
        name = preferred[index % len(preferred)]
        day = monday + timedelta(days=WEEKDAYS.index(name))
        # A second post on the same preferred day rolls forward a week's worth
        # of days rather than colliding.
        if index >= len(preferred):
            day += timedelta(days=1)
        slots.extend(daily_slots(day, schedule, platform, tz, rng, count=1))
    return sorted(slots, key=lambda slot: slot.when)


def describe(slots: dict[Platform, list[Slot]]) -> str:
    total = sum(len(items) for items in slots.values())
    parts = [f"{len(items)} {platform.value}" for platform, items in slots.items() if items]
    return f"{total} slots: " + ", ".join(parts)
