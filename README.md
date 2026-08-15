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
| 2. Generation: providers, prompts, ideas, drafts | next |
| 3. Diagrams: flowcharts for LinkedIn, visuals for X where they help | next |
| 4. Evaluation: authenticity, slop, repetition, monetization | planned |
| 5. Learning from your edits | planned |
| 6. Scheduling and Excel export (usable MVP) | planned |
| 7. Research layer | planned |
| 8. Dashboard | planned |
| 9. Performance tracking and platform integration | planned |

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
