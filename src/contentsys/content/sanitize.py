"""Post-generation guarantees.

Everything here runs on model output before it is allowed to become a draft.
These are checks the prompt also asks for, deliberately duplicated in code,
because a prompt instruction is a strong suggestion and some of these need to
be guarantees.

Two of them matter more than the rest.

**No em dashes.** The single clearest surface tell of machine-written text.
The prompt asks; this enforces.

**No invented experiences.** A draft may not claim a first-person experience
unless it carries a verified ``experience_id``. This is the invariant the
whole product rests on: a system that fabricates a job, a project or a
conversation is worse than no system, because the failure is invisible to
everyone except the person it is lying about.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

EM_DASH = chr(0x2014)
HORIZONTAL_BAR = chr(0x2015)
EN_DASH = chr(0x2013)

#: Rewrites applied silently. A comma carries the same pause without the tell.
_PUNCTUATION_FIXES: tuple[tuple[str, str], ...] = (
    (f" {EM_DASH} ", ", "),
    (f" {HORIZONTAL_BAR} ", ", "),
    (f" {EN_DASH} ", ", "),
    (f"{EM_DASH}", ", "),
    (f"{HORIZONTAL_BAR}", ", "),
)

#: First-person claims about having done, built, worked on or experienced
#: something. Matching one of these means the draft is asserting autobiography
#: and must be backed by a verified experience.
#:
#: Tuned to catch claims about the past and about work, and to leave alone the
#: present-tense thinking that is always safe to say: "i think", "i find this
#: interesting", "i keep coming back to". Those are opinions, not history.
_EXPERIENCE_CLAIMS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhen i (?:was|were|used to|first)\b", re.I),
    re.compile(r"\bi (?:was|used to be) (?:working|building|at|an?|the)\b", re.I),
    re.compile(r"\bi (?:built|shipped|wrote|launched|designed|implemented|deployed)\b", re.I),
    re.compile(r"\bi (?:worked|interned|consulted|contributed)\b", re.I),
    re.compile(r"\bi (?:spent|took) (?:\w+ ){0,3}(?:months?|weeks?|years?|days?)\b", re.I),
    re.compile(r"\b(?:my|our) (?:team|manager|client|company|colleague|coworker)\b", re.I),
    re.compile(
        r"\bin my (?:last|previous|first|current) (?:job|role|team|company|internship)\b", re.I
    ),
    re.compile(r"\ba (?:few|couple of) (?:years?|months?) ago,? i\b", re.I),
    re.compile(r"\bi once\b", re.I),
    re.compile(r"\bi remember (?:when|the time)\b", re.I),
)

#: Claims of measured outcomes. A fabricated number is the most damaging kind
#: of invention because it is the most quotable.
_METRIC_CLAIMS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi (?:cut|reduced|improved|increased|grew|saved|shipped)\b.{0,40}\d", re.I),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:x|%|percent)\b.{0,30}\b(?:faster|slower|better|improvement)\b", re.I
    ),
)


@dataclass
class SanitizeResult:
    """What sanitising did, and what it could not fix.

    ``text`` is always safe to show. ``violations`` is non-empty when the
    draft must be rejected rather than published: those are things a rewrite
    cannot fix, because the problem is the claim, not the wording.
    """

    text: str
    changed: bool = False
    repairs: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def strip_banned_punctuation(text: str) -> tuple[str, bool]:
    """Remove em dashes and their lookalikes.

    Returns the cleaned text and whether anything changed. An unspaced en
    dash is left alone: it is a legitimate numeric range.
    """
    if not any(bad in text for bad, _ in _PUNCTUATION_FIXES):
        # Return early rather than running the tidy-up passes over text that
        # never contained a dash. Those passes exist to clean up after this
        # function's own replacements, and letting them touch untouched text
        # is how a sanitiser starts quietly editing things nobody asked it to.
        return text, False

    cleaned = text
    for bad, replacement in _PUNCTUATION_FIXES:
        cleaned = cleaned.replace(bad, replacement)
    # Collapse the double punctuation a replacement can leave behind, for
    # example ", ," or ",." where the dash sat next to an existing comma.
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r",\s*([.!?])", r"\1", cleaned)
    return cleaned, cleaned != text


def find_experience_claims(text: str) -> list[str]:
    """Return the first-person experience claims a draft makes.

    Used two ways: to reject a draft that invents autobiography, and to check
    that a draft written from a real experience is actually claiming the thing
    that experience supports.
    """
    found: list[str] = []
    for pattern in (*_EXPERIENCE_CLAIMS, *_METRIC_CLAIMS):
        for match in pattern.finditer(text):
            found.append(match.group(0).strip())
    return found


def normalise_whitespace(text: str) -> str:
    """Tidy without flattening.

    Deliberately conservative. Double spaces after a full stop and stretched
    letters are habits worth keeping, so only trailing whitespace and runs of
    three or more blank lines are touched.
    """
    text = unicodedata.normalize("NFC", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_wrapper_quotes(text: str) -> str:
    """Remove quotes a model sometimes wraps the whole post in.

    Only when they enclose the entire string and nothing inside is quoted,
    so a post that legitimately opens and closes on dialogue is left alone.
    """
    stripped = text.strip()
    for opening, closing in (('"', '"'), ("'", "'"), (chr(0x201C), chr(0x201D))):
        if len(stripped) > 2 and stripped.startswith(opening) and stripped.endswith(closing):
            inner = stripped[1:-1]
            if opening not in inner and closing not in inner:
                return inner.strip()
    return stripped


def sanitize(
    text: str,
    *,
    has_verified_experience: bool = False,
    max_length: int | None = None,
) -> SanitizeResult:
    """Clean a generated draft and check what cannot be cleaned.

    ``has_verified_experience`` says whether this draft is linked to a real
    row in the experiences table. When it is not, any first-person experience
    claim is a fabrication and becomes a violation rather than a repair,
    because rewriting it would just launder the invention into vaguer
    language.
    """
    result = SanitizeResult(text=text)

    cleaned = strip_wrapper_quotes(text)
    if cleaned != text.strip():
        result.repairs.append("removed wrapper quotes")

    cleaned, punctuation_changed = strip_banned_punctuation(cleaned)
    if punctuation_changed:
        result.repairs.append("replaced em dashes")

    tidied = normalise_whitespace(cleaned)
    if tidied != cleaned:
        result.repairs.append("normalised whitespace")
    cleaned = tidied

    result.text = cleaned
    result.changed = cleaned != text

    if not has_verified_experience:
        claims = find_experience_claims(cleaned)
        if claims:
            quoted = ", ".join(sorted(set(claims))[:3])
            result.violations.append(
                f"claims a personal experience that is not in the knowledge base: {quoted}"
            )

    if max_length is not None and len(cleaned) > max_length:
        result.violations.append(
            f"too long: {len(cleaned)} characters against a limit of {max_length}"
        )

    if not cleaned:
        result.violations.append("empty after sanitising")

    return result
