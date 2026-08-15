"""Deciding what a diagram should say.

The model's only job here is structure: which shape the argument takes, what
the steps are, and which one is the interesting step. It never chooses a
colour, a position or a font, because that is where models are weak and code
is strong.

It is also allowed to say no. A post that is a one line opinion has no
structure to draw, and a diagram attached to it is decoration that cost a
generation call and a review decision.
"""

from __future__ import annotations

from contentsys.config import Platform
from contentsys.llm.base import LLMError, LLMProvider, LLMRequest
from contentsys.prompts import PromptContext, build_system
from contentsys.visuals.spec import DIAGRAM_SCHEMA, DiagramSpec, SpecError

DIAGRAM_BRIEF = """\
Design a diagram for the post below.

Pick the shape that matches the argument the post is actually making:

- chain: a sequence of reductions, each step trading the problem for an
  easier one. This is the most common shape for this account's writing.
- flow: a protocol with an exchange between two parties, or rounds.
- comparison: two systems side by side on the dimensions that actually
  differ. Do not compare on dimensions where they agree, except where the
  agreement is the point.
- timeline: an ordered sequence of events or rounds.

Rules that come from where these are seen, which is a phone in a scrolling
timeline:

- At most 5 steps. Fewer is better. If the argument needs 7, it needs two
  diagrams or a shorter argument.
- A label is 1 to 4 words. A note is a short phrase, not a sentence.
- Every note should say what that step buys or costs. "R1CS" is a label,
  "every operation as an arithmetic constraint" is a note. A note that just
  restates the label is wasted space.
- Highlight exactly one step, the one where the interesting thing happens.
  Or highlight none. Highlighting several highlights nothing.
- The title is the claim, not the topic. "How Spartan verifies millions of
  constraints" is a title. "Spartan" is not.

Write alt_text describing the diagram for someone who cannot see it. This is
required.

Do not invent a step that is not in the post. If the post does not actually
contain a structure, that is a real answer and you should say so rather than
manufacture one.\
"""


def diagram_request(
    context: PromptContext,
    *,
    content: str,
    model: str | None = None,
    effort: str | None = None,
) -> LLMRequest:
    return LLMRequest(
        # Reuses the same cached prefix as generation, so asking for a diagram
        # after drafting a post costs a cache read rather than a fresh one.
        system=build_system(context),
        prompt=f"{DIAGRAM_BRIEF}\n\n---\n{content.strip()}\n---",
        max_tokens=2048,
        model=model,
        effort=effort,
        json_schema=DIAGRAM_SCHEMA,
        tags=("diagram", context.platform.value),
    )


def wants_diagram(context: PromptContext, content_type: str) -> bool:
    """Whether this post is eligible for one at all.

    Reads the owner's configured policy rather than deciding here: every
    LinkedIn post gets one, and on X only the structural content types do.
    """
    return context.rules.wants_visual(context.platform, content_type)


def generate_diagram(
    provider: LLMProvider,
    context: PromptContext,
    *,
    content: str,
    model: str | None = None,
    effort: str | None = None,
) -> DiagramSpec | None:
    """Produce a spec for a post, or None when a diagram would not help.

    Returns None rather than raising on a malformed or structureless answer.
    A missing diagram is a small loss; a failed weekly run over one bad JSON
    response is not.
    """
    request = diagram_request(context, content=content, model=model, effort=effort)
    try:
        payload = provider.complete_json(request)
    except (LLMError, ValueError):
        return None

    try:
        spec = DiagramSpec.from_dict(payload).truncated()
        spec.validate()
    except SpecError:
        return None
    return spec


def diagram_path(root, platform: Platform, identifier: str):
    """Where a rendered diagram lives.

    Grouped by platform because the two get reviewed at different cadences,
    and a week of X images should not bury the two LinkedIn ones.
    """
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in identifier)[:60]
    return root / "diagrams" / platform.value.lower() / f"{safe}.png"
