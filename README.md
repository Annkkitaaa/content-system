# contentsys

A personal content system. It turns your own knowledge, opinions and
experiences into X and LinkedIn drafts that sound like you, produces a full
week in one batch, and learns from how you edit them.

It is not a social media post generator. The difference matters: the target is
not "this is a good post", it is "a human reading this would believe this
person actually had this thought".

## What it does

Run one command once a week. It reads your voice profile, your knowledge base
and your recent content history, generates a pool of ideas, drafts the
strongest ones, scores every draft, regenerates the weak ones, schedules them
across the week, and writes an Excel workbook.

You open the workbook, edit what you want to change, mark what you approve,
and post through the week. Your edits feed back in, so the next batch sounds
more like you than the last one.

Default weekly target: 70 X posts (10 a day) and 2 LinkedIn posts.

## What it will not do

- Invent an experience you did not have. Personal content is generated only
  from verified rows in your knowledge base, and drafts claiming otherwise are
  rejected in code before they reach you.
- Publish anything on its own. Approval is manual by default and stays that
  way unless you deliberately turn something else on.
- Write engagement bait. Under the X Original Content Rewards program that is
  a violation, not a style choice, so a bait verdict fails a draft outright.
- Use em dashes. Anywhere.

## Status

Under construction, one phase per pull request.

| Phase | State |
|---|---|
| 0. Foundation: config, provider protocol, tooling | done |
| 1. Knowledge base, sample ingestion, voice profile | done |
| 2. Providers, prompts, guarantees | done |
| 3. Idea and draft engines, the eight modes | done |
| 4. Diagrams: flowcharts for LinkedIn, visuals for X where they help | next |
| 5. Evaluation: slop, repetition, monetization, voice | done |
| 6. Learning from your edits | done |
| 7. Scheduling and Excel export (usable MVP) | next |
| 8. Research layer | planned |
| 9. Dashboard | planned |
| 10. Performance tracking and platform integration | planned |

## Generating

Nothing here publishes. Every command prints drafts for you to judge.

```bash
contentsys generate daily -n 10
contentsys generate linkedin
contentsys generate ideas "sumcheck"
contentsys generate explain "polynomial commitments"
contentsys generate react "a bridge was drained overnight"
contentsys generate personal
```

Brain dump is the one worth knowing about. Give it a messy thought and it
cleans it up just enough to post, without adding a hook, a lesson, or a
conclusion you did not write:

```bash
contentsys generate dump "zk confused me for ages then i realised you're not proving the secret, you're proving something satisfying the constraints exists"
```

Add `--provider mock` to any of these to see the pipeline run offline and for
free.

## Teaching it

Rewrite a draft the way you actually wanted it, then show it both versions:

```bash
contentsys teach draft.txt edited.txt
```

It works out what changed (shorter, lowercased, hook cut, hedging removed) and
remembers it. A preference is recorded the first time but not used. It only
reaches the prompt once the same change appears repeatedly, so one unusual
edit cannot permanently reshape how everything is written.

Contradicting evidence decays a preference rather than fighting it, which
means a genuine reversal takes as long to learn as the original did. If it
ever learns something wrong:

```bash
contentsys forget prefers_lowercase
```

## Install

Requires Python 3.12 or newer.

```bash
python -m venv .venv
```

Then activate it. On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install the package with the development extras:

```bash
pip install -e ".[dev]"
```

Add the backend you plan to use. On a Claude subscription:

```bash
pip install -e ".[agent,dev]"
```

Or with an Anthropic API key:

```bash
pip install -e ".[api,dev]"
```

## Configure

```bash
copy .env.example .env
```

Every setting has a working default, so an empty `.env` runs. The two worth
checking first are `CONTENTSYS_PROVIDER` and `CONTENTSYS_TIMEZONE`.

Structured configuration lives in `config/`:

- `config/schedule.yaml` sets cadence, posting windows and timing jitter
- `config/content_mix.yaml` sets the target spread of content types

Confirm what the system actually resolved:

```bash
contentsys config
```

## Model access

Claude Pro does not include Anthropic API access; those are separate products
with separate billing. It does support the Claude Agent SDK, which runs on
your subscription with no API key, and that is the default provider here.

The Messages API provider is also available and is the better choice at
volume: it supports prompt caching, which matters when a weekly run reuses the
same large voice-profile prefix across roughly a hundred calls.

Switching is one line in `.env`. Nothing else in the system changes.

## Develop

```bash
pytest
ruff check .
python tools/check_style.py
```

Conventions and the invariants that must not be broken are in
[CLAUDE.md](CLAUDE.md). The architecture is in
[docs/architecture.md](docs/architecture.md).

## Privacy

Your knowledge base is personal data. `data/`, `exports/`, `.env` and every
`.xlsx` are gitignored. Nothing about you is committed to this repository.
