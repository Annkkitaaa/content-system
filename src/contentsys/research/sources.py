"""Fetching what actually happened.

Three public sources, none of which need a key: Hacker News through the
Algolia index, arXiv's Atom API, and GitHub releases. The shapes below were
checked against live responses rather than assumed.

Everything here returns a :class:`Finding`, which is deliberately a thin
record of an external fact: a title, a date, a link, and nothing resembling an
opinion. Interpretation happens later and somewhere else. That separation is
the point of this package, because the failure it prevents is a source fact
quietly becoming something the owner did or believes.

Network failure is never fatal. A source that times out is skipped and the run
continues, because a weekly batch that dies because Hacker News was slow is
worse than one that draws on two sources instead of three.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

USER_AGENT = "contentsys/0.1 (personal content system)"
TIMEOUT = 20.0


@dataclass
class Finding:
    """One external fact.

    ``summary`` is whatever the source said, not what anyone makes of it. The
    distinction is enforced by this type carrying no field in which an opinion
    could be stored.
    """

    title: str
    source: str
    url: str | None = None
    published: datetime | None = None
    summary: str = ""
    #: Source-specific signal of attention, such as Hacker News points.
    weight: float = 0.0
    tags: list[str] = field(default_factory=list)

    @property
    def age_days(self) -> float:
        if self.published is None:
            return 999.0
        return (datetime.now(UTC) - self.published).total_seconds() / 86400

    def key(self) -> str:
        """Identity for deduplication, since sources overlap."""
        return re.sub(r"[^a-z0-9]+", " ", self.title.lower()).strip()


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read().decode("utf-8", "replace")


def _safe(fetcher, label: str) -> list[Finding]:
    """Run a fetcher, returning nothing rather than raising.

    A weekly batch that dies because one source was slow is worse than one
    drawing on two sources instead of three.
    """
    try:
        return fetcher()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        return []


def hacker_news(queries: list[str], *, limit: int = 12, min_points: int = 5) -> list[Finding]:
    """Recent stories matching each query.

    Queries are searched separately so one busy topic cannot crowd the others
    out of the result set.
    """
    findings: list[Finding] = []
    for query in queries:
        params = urllib.parse.urlencode(
            {
                "tags": "story",
                "query": query,
                "hitsPerPage": max(1, limit // max(1, len(queries)) + 2),
            }
        )

        def fetch(params: str = params, query: str = query) -> list[Finding]:
            payload = json.loads(_get(f"https://hn.algolia.com/api/v1/search_by_date?{params}"))
            found: list[Finding] = []
            for hit in payload.get("hits", []):
                title = (hit.get("title") or "").strip()
                points = hit.get("points") or 0
                if not title or points < min_points:
                    continue
                found.append(
                    Finding(
                        title=title,
                        source="Hacker News",
                        url=hit.get("url")
                        or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                        published=_parse_iso(hit.get("created_at")),
                        weight=float(points),
                        tags=[query],
                    )
                )
            return found

        findings.extend(_safe(fetch, "hacker news"))
    return findings


def arxiv(categories: list[str], keywords: list[str], *, limit: int = 12) -> list[Finding]:
    """Papers matching the configured keywords, newest first.

    Searches the abstract rather than fetching recent listings and filtering
    them. That distinction is the difference between working and not: cs.CR is
    dominated by machine learning security, so a title filter over the most
    recent few dozen submissions returned nothing at all. Asking arXiv to
    search returns the relevant papers directly.

    The query is URL encoded rather than concatenated, because the terms
    contain spaces and quotes and a hand-built query string raises
    InvalidURL before it ever reaches the network.
    """

    def fetch() -> list[Finding]:
        terms = [f'abs:"{keyword}"' if " " in keyword else f"abs:{keyword}" for keyword in keywords]
        clauses = []
        if terms:
            clauses.append("(" + " OR ".join(terms) + ")")
        if categories:
            clauses.append("(" + " OR ".join(f"cat:{category}" for category in categories) + ")")
        if not clauses:
            return []

        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {
                "search_query": " AND ".join(clauses),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": limit * 2,
            }
        )
        entries = re.findall(r"<entry>(.*?)</entry>", _get(url), re.S)
        found: list[Finding] = []

        for entry in entries:
            title = _tag(entry, "title")
            if not title:
                continue
            found.append(
                Finding(
                    title=title,
                    source="arXiv",
                    url=_tag(entry, "id"),
                    published=_parse_iso(_tag(entry, "published")),
                    summary=_tag(entry, "summary")[:600],
                    weight=1.0,
                    tags=["paper"],
                )
            )
            if len(found) >= limit:
                break
        return found

    return _safe(fetch, "arxiv")


def github_releases(repositories: list[str], *, limit: int = 12) -> list[Finding]:
    """Recent releases from protocol and prover repositories."""
    findings: list[Finding] = []
    for repo in repositories:

        def fetch(repo: str = repo) -> list[Finding]:
            payload = json.loads(_get(f"https://api.github.com/repos/{repo}/releases?per_page=2"))
            found: list[Finding] = []
            for release in payload:
                if release.get("draft") or release.get("prerelease"):
                    continue
                tag = release.get("tag_name") or ""
                name = (release.get("name") or "").strip()
                title = f"{repo} released {tag}" + (f": {name}" if name and name != tag else "")
                found.append(
                    Finding(
                        title=title,
                        source="GitHub",
                        url=release.get("html_url"),
                        published=_parse_iso(release.get("published_at")),
                        summary=(release.get("body") or "")[:600],
                        weight=1.0,
                        tags=["release", repo.split("/")[-1]],
                    )
                )
            return found

        findings.extend(_safe(fetch, f"github {repo}"))
        if len(findings) >= limit:
            break
    return findings[:limit]


def _tag(xml: str, name: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", xml, re.S)
    return " ".join(match.group(1).split()) if match else ""


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def recent(findings: list[Finding], *, max_age_days: int) -> list[Finding]:
    """Keep only what is actually recent.

    Reacting to something three months old reads as manufacturing relevance,
    which is the failure this whole layer is supposed to avoid.
    """
    cutoff = timedelta(days=max_age_days).total_seconds() / 86400
    return [finding for finding in findings if finding.age_days <= cutoff]


def deduplicate(findings: list[Finding]) -> list[Finding]:
    """Collapse the same story appearing in more than one source."""
    seen: dict[str, Finding] = {}
    for finding in findings:
        key = finding.key()
        existing = seen.get(key)
        if existing is None or finding.weight > existing.weight:
            seen[key] = finding
    return list(seen.values())


def rank(findings: list[Finding]) -> list[Finding]:
    """Most worth reacting to first.

    Attention is scored on a log scale and recency linearly, so a quiet story
    from yesterday outranks a busy one from last week.

    The log matters. Hacker News points span three orders of magnitude, so a
    linear term let a single 500 point story from nine days ago dominate
    everything fresh, which is precisely the item least worth reacting to: by
    the time this week's posts go out the conversation is over.
    """

    def score(finding: Finding) -> float:
        attention = math.log10(1.0 + max(0.0, finding.weight)) * 2.0
        freshness = max(0.0, 10.0 - finding.age_days) * 2.0
        return -(attention + freshness)

    return sorted(findings, key=lambda finding: (score(finding), finding.title))
