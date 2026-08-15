"""Getting the owner's material into the knowledge base.

Two entry points. :func:`load_seed` reads the committed YAML under ``seed/``,
which is the public starting set. :func:`import_samples` reads whatever the
owner exports later: a plain text file, a Markdown file, a CSV, or the JSON
from an X archive.

Every path is idempotent. Re-importing an archive that overlaps an earlier one
must not double count anything, because the voice profile is a statistical
measurement and duplicates quietly bias it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlmodel import Session, select

from contentsys.config import PROJECT_ROOT
from contentsys.db.models import (
    Confidence,
    Experience,
    KnowledgeItem,
    Opinion,
    SampleSource,
    Topic,
    WritingSample,
)

SEED_DIR = PROJECT_ROOT / "seed"

#: Posts split by a blank line when importing free text.
_BLANK_LINE = re.compile(r"\n\s*\n")


@dataclass
class ImportReport:
    """What an import actually did.

    Reporting skipped counts matters: an import that says "imported 0" when
    everything was a duplicate looks like a failure, and an import that says
    "imported 400" when 380 were duplicates silently corrupts the profile.
    """

    added: int = 0
    skipped_duplicate: int = 0
    skipped_empty: int = 0

    def describe(self) -> str:
        parts = [f"{self.added} added"]
        if self.skipped_duplicate:
            parts.append(f"{self.skipped_duplicate} already present")
        if self.skipped_empty:
            parts.append(f"{self.skipped_empty} empty or too short")
        return ", ".join(parts)


def fingerprint(text: str) -> str:
    """A stable id for a piece of writing.

    Normalises whitespace and case so a re-export with different line wrapping
    is recognised as the same post, but keeps the words themselves, so two
    genuinely different posts never collide.
    """
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


def _coerce_date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value if not isinstance(value, datetime) else value.date()
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], pattern).date()
        except ValueError:
            continue
    return None


def add_sample(
    session: Session,
    *,
    content: str,
    source: SampleSource,
    published_at: date | None = None,
    url: str | None = None,
    topic: str | None = None,
    impressions: int | None = None,
    excluded: bool = False,
    exclusion_reason: str | None = None,
    min_length: int = 2,
) -> WritingSample | None:
    """Store one sample. Returns None when it is a duplicate or too short."""
    text = content.strip()
    if len(text) < min_length:
        return None

    digest = fingerprint(text)
    existing = session.exec(
        select(WritingSample).where(WritingSample.fingerprint == digest)
    ).first()
    if existing is not None:
        return None

    sample = WritingSample(
        source=source,
        content=text,
        fingerprint=digest,
        published_at=published_at,
        url=url,
        topic=topic,
        impressions=impressions,
        excluded=excluded,
        exclusion_reason=exclusion_reason,
    )
    session.add(sample)
    return sample


def load_seed(session: Session, seed_dir: Path | None = None) -> dict[str, ImportReport]:
    """Load the committed starting knowledge base.

    Safe to run more than once. Experiences, opinions, knowledge items and
    topics are matched on their natural key so a second run updates rather
    than duplicates.
    """
    directory = seed_dir or SEED_DIR
    reports = {
        "samples": ImportReport(),
        "experiences": ImportReport(),
        "opinions": ImportReport(),
        "knowledge": ImportReport(),
        "topics": ImportReport(),
    }

    samples_file = directory / "samples.yaml"
    if samples_file.exists():
        data = yaml.safe_load(samples_file.read_text(encoding="utf-8")) or {}
        for source_name, entries in data.items():
            try:
                source = SampleSource(source_name)
            except ValueError:
                source = SampleSource.OTHER
            for entry in entries or []:
                added = add_sample(
                    session,
                    content=entry.get("content", ""),
                    source=source,
                    published_at=_coerce_date(entry.get("published_at")),
                    url=entry.get("url"),
                    topic=entry.get("topic"),
                    impressions=entry.get("impressions"),
                    excluded=bool(entry.get("exclude", False)),
                    exclusion_reason=entry.get("exclusion_reason"),
                )
                if added is None:
                    reports["samples"].skipped_duplicate += 1
                else:
                    reports["samples"].added += 1

    profile_file = directory / "profile.yaml"
    if profile_file.exists():
        data = yaml.safe_load(profile_file.read_text(encoding="utf-8")) or {}

        for entry in data.get("experiences") or []:
            summary = entry.get("summary", "").strip()
            if not summary:
                continue
            existing = session.exec(select(Experience).where(Experience.summary == summary)).first()
            if existing is not None:
                reports["experiences"].skipped_duplicate += 1
                continue
            session.add(
                Experience(
                    summary=summary,
                    detail=entry.get("detail"),
                    organisation=entry.get("organisation"),
                    role=entry.get("role"),
                    started_on=_coerce_date(entry.get("started_on")),
                    ended_on=_coerce_date(entry.get("ended_on")),
                    evidence=entry.get("evidence"),
                    evidence_url=entry.get("evidence_url"),
                    confidence=Confidence(entry.get("confidence", "stated")),
                    tags=entry.get("tags") or [],
                )
            )
            reports["experiences"].added += 1

        for entry in data.get("opinions") or []:
            statement = entry.get("statement", "").strip()
            if not statement:
                continue
            existing = session.exec(select(Opinion).where(Opinion.statement == statement)).first()
            if existing is not None:
                reports["opinions"].skipped_duplicate += 1
                continue
            session.add(
                Opinion(
                    statement=statement,
                    reasoning=entry.get("reasoning"),
                    topic=entry.get("topic"),
                    strength=entry.get("strength", "held"),
                    evidence=entry.get("evidence"),
                    confidence=Confidence(entry.get("confidence", "stated")),
                )
            )
            reports["opinions"].added += 1

        for entry in data.get("knowledge") or []:
            concept = entry.get("concept", "").strip()
            if not concept:
                continue
            existing = session.exec(
                select(KnowledgeItem).where(KnowledgeItem.concept == concept)
            ).first()
            if existing is not None:
                reports["knowledge"].skipped_duplicate += 1
                continue
            session.add(
                KnowledgeItem(
                    concept=concept,
                    domain=entry.get("domain", "general"),
                    depth=entry.get("depth", "familiar"),
                    notes=entry.get("notes"),
                    evidence=entry.get("evidence"),
                    evidence_url=entry.get("evidence_url"),
                    tags=entry.get("tags") or [],
                )
            )
            reports["knowledge"].added += 1

        for entry in data.get("topics") or []:
            name = entry.get("name", "").strip()
            if not name:
                continue
            existing = session.exec(select(Topic).where(Topic.name == name)).first()
            if existing is not None:
                reports["topics"].skipped_duplicate += 1
                continue
            session.add(
                Topic(
                    name=name,
                    is_core=bool(entry.get("core", True)),
                    cooldown_days=int(entry.get("cooldown_days", 5)),
                    avoid=bool(entry.get("avoid", False)),
                    avoid_reason=entry.get("avoid_reason"),
                )
            )
            reports["topics"].added += 1

    return reports


def _texts_from_file(path: Path) -> Iterable[dict[str, Any]]:
    """Pull posts out of whatever format the owner exported.

    Supports the four things an export realistically is: a text or Markdown
    file with posts separated by blank lines, a CSV with a text column, or the
    JSON an X archive produces.
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        raw = path.read_text(encoding="utf-8")
        # An X archive prefixes its JSON with a JavaScript assignment.
        raw = re.sub(r"^\s*window\.[\w.]+\s*=\s*", "", raw).strip().rstrip(";")
        payload = json.loads(raw)
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            # Archives nest the post one level down under "tweet".
            item = record.get("tweet", record) if isinstance(record, dict) else {}
            text = item.get("full_text") or item.get("text") or item.get("content")
            if not text:
                continue
            yield {
                "content": text,
                "published_at": _coerce_date(item.get("created_at")),
                "url": item.get("url"),
            }
        return

    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                lowered = {(k or "").strip().lower(): v for k, v in row.items()}
                text = (
                    lowered.get("text")
                    or lowered.get("content")
                    or lowered.get("post")
                    or lowered.get("tweet")
                )
                if not text:
                    continue
                yield {
                    "content": text,
                    "published_at": _coerce_date(lowered.get("date") or lowered.get("created_at")),
                    "url": lowered.get("url") or lowered.get("link"),
                    "impressions": _as_int(lowered.get("impressions")),
                }
        return

    for block in _BLANK_LINE.split(path.read_text(encoding="utf-8")):
        text = block.strip()
        if text:
            yield {"content": text}


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def import_samples(
    session: Session,
    path: Path,
    source: SampleSource,
    *,
    min_length: int = 2,
) -> ImportReport:
    """Import writing samples from a file."""
    report = ImportReport()
    for record in _texts_from_file(path):
        content = (record.get("content") or "").strip()
        if len(content) < min_length:
            report.skipped_empty += 1
            continue
        added = add_sample(
            session,
            content=content,
            source=source,
            published_at=record.get("published_at"),
            url=record.get("url"),
            impressions=record.get("impressions"),
            min_length=min_length,
        )
        if added is None:
            report.skipped_duplicate += 1
        else:
            report.added += 1
    return report
