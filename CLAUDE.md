# Repository conventions

## The one hard rule: no em dashes

Never use an em dash, a horizontal bar, or a spaced en dash. Anywhere. Source,
comments, docstrings, docs, commit messages, prompts, generated content.

Use a comma, a full stop, a colon, or parentheses instead.

This is enforced four ways: an instruction in every generation prompt, a
runtime sanitizer over model output, `tools/check_style.py` over the
repository, and CI. Run `python tools/check_style.py --fix` before committing.

The reason is not taste. The em dash is one of the clearest surface tells of
machine-written text, and the entire point of this project is content that
does not read that way. A rule the codebase itself breaks is a rule the model
will break too.

## What this project is

A personal content system. It turns the owner's real knowledge, opinions and
experiences into X and LinkedIn drafts, in one weekly batch, and learns from
how those drafts get edited.

The measure of success is not "is this a good post". It is "would a human
reading this believe this person actually had this thought". When those two
pull in different directions, the second one wins.

## Invariants that are not negotiable

**Never invent an experience.** Content may not assert a first-person
experience unless it is backed by a row in the `experiences` table with a real
`experience_id`. This is a code-level check with adversarial tests, not a
prompt instruction, because prompt instructions are suggestions and this is
not. When there is no verified experience to draw on, the generator writes an
analytical or educational post instead.

**Keep the three layers separate.**

| Layer | Package | Holds |
|---|---|---|
| Identity | `knowledge/` | who the owner is, what they know, what they think |
| External | `research/` | what happened in the world, with sources |
| Generated | `content/` | interpretation built from the other two |

A source fact never becomes a personal experience by passing through a
prompt. Anything in `content/` that claims otherwise is a bug.

**Nothing publishes without explicit approval.** The default path is generate,
review, edit, approve, publish. Any automation added later starts switched
off.

**Engagement is last.** The priority order is authenticity, quality,
originality, usefulness, consistency, and only then engagement. A post that is
genuinely interesting and reaches fifty people beats a fake one that reaches
fifty thousand.

## X monetization constraints

The account is being built toward the X Original Content Rewards program, so
some rules are program compliance rather than style preference:

- Engagement-bait calls to action are a **program violation**. The
  monetization evaluator fails such a draft outright rather than scoring it
  down. Do not soften this into a warning.
- Content must be original. When a post is commentary on someone else's work,
  it has to add substantive analysis, and the provenance is recorded per post.
- Posting timing carries deliberate jitter. Machine-regular cadence is a
  demotion signal.
- Ranking rewards conversation depth and dwell time, so "does this give a
  reader something specific to reply to" is a real quality signal here, and it
  is measured separately from bait.

## Code conventions

- Python 3.12, `from __future__ import annotations` at the top of every module.
  The one exception is `db/models.py`: SQLModel resolves `Relationship`
  targets through SQLAlchemy's class registry rather than through `typing`,
  and PEP 563 turns a forward reference into a string that registry cannot
  parse. Do not "fix" it.
- Type hints everywhere. Prefer `X | None` over `Optional[X]`.
- Pydantic models for anything crossing a boundary (config, LLM payloads, API).
- `ruff check .` and `ruff format` clean before commit.
- Comments explain why, not what. If a line needs a comment to say what it
  does, rewrite the line.

## Layout

```
config/            YAML for schedule, content mix, slop rules
src/contentsys/
  config.py        settings and thresholds
  db/              SQLModel tables and migrations
  knowledge/       identity layer
  voice/           voice profile and voice memory
  llm/             provider protocol and implementations
  prompts/         modular prompt fragments and the composer
  content/         idea and generation engines, the eight modes
  evaluation/      authenticity, voice, slop, repetition, accuracy, monetization
  research/        external sources
  scheduling/      weekly slot assignment
  export/          Excel workbook
  pipeline/        the weekly run orchestrator
tests/fixtures/    the regression corpus
tools/             repo tooling
```

## Testing

`pytest` must pass before any PR. The fixture corpus in `tests/fixtures/`
holds labelled examples (good, bad, slop, authentic, too generic, too
promotional, too technical) asserted against the evaluators, so scoring
cannot drift silently as prompts change.

The whole pipeline runs against `MockProvider`, so an end to end weekly run is
testable offline and for free.

## Secrets

`.env` is gitignored. `.env.example` carries the shape with empty values. No
key, token or personal note ever goes into a tracked file. The knowledge base
is personal data: `data/` and `exports/` are gitignored and stay that way.
