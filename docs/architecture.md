# Architecture

## The problem this solves

Generic AI content is easy to produce and worthless. The hard part is content
that a reader would believe you actually thought.

That framing drives most of the design. The system is not optimised to write
good posts; it is optimised to write posts that are recognisably yours and
that do not claim anything about you that is not true. Where those goals
conflict with reach, they win.

## Data flow

```
       knowledge (who I am)              research (what happened)
               |                                    |
               v                                    v
 samples --> VOICE ENGINE --------> PROMPT COMPOSER <--- source facts
               |                          |
               |                          v
               |                    IDEA ENGINE ---> idea pool
               |                          |
               |                          v
               +----------------> GENERATION ENGINE ---> draft
                                          |
                                          v
                    EVALUATION: authenticity, voice match, slop,
                    repetition, technical accuracy, monetization
                                          |
                             pass? --no--> regenerate (bounded)
                                          | yes
                                          v
                                    SCHEDULER ---> EXCEL WORKBOOK
                                          |
                                          v
                              you edit / approve / publish
                                          |
                                          v
                          EDIT DIFF ---> VOICE MEMORY (loops back)
```

## The three layers, and why they never merge

| Layer | Package | Holds | Never |
|---|---|---|---|
| Identity | `knowledge/` | what you know, think, have done | invents anything |
| External | `research/` | what happened, with a source | becomes your experience |
| Generated | `content/` | interpretation built from both | asserts an unbacked claim |

The most common failure mode in systems like this is a source fact quietly
becoming a personal anecdote somewhere in a prompt. Keeping the layers in
separate packages with an explicit boundary makes that a type error rather
than a judgement call.

### The experience invariant

Content may not assert a first-person experience unless it carries a real
`experience_id` from the `experiences` table. Personal-mode generation
retrieves those rows and the composed prompt names exactly which experiences
are available. Every draft is then post-checked for first-person experience
claims, and one without a matching id fails validation.

This is deliberately implemented in code with adversarial tests rather than as
a prompt instruction. Prompt instructions are strong suggestions. This needs
to be a guarantee.

When the knowledge base has nothing relevant, the generator does not
improvise. It writes an analytical or educational post instead, or drops the
slot.

## Components

### Voice engine

Two layers, because voice is not formatting.

The **surface layer** measures mechanics from your samples: sentence length
distribution, lowercase ratio, punctuation habits, contractions, fragments,
emoji use, how you open, how you end. This is cheap to compute and easy to
check after the fact.

The **semantic layer** is an extracted profile of how you actually think:
how you build an argument, where you hedge and where you commit, what you
find interesting about a topic, what you refuse to claim, what your humour
is made of.

The semantic layer drives generation. The surface layer is a post-generation
check. Getting this the wrong way round produces text that counts lowercase
letters correctly and still sounds like nobody.

Both persist as structured JSON, not one enormous prompt string, so they can
evolve field by field.

### Voice memory

When you mark an edit as worth learning from, the system diffs your version
against the draft and classifies what changed: a hook removed, a claim
softened, something lowercased, a sentence cut, an abstraction replaced with a
concrete example. Each classification becomes a `voice_preference` row whose
confidence increments when the same pattern shows up again.

Only preferences above a confidence floor get injected into future prompts.
That floor exists so a single unusual edit does not permanently distort the
model of how you write.

### Idea engine

Ideas are generated before posts, and separately. An idea carries topic,
angle, why it is interesting, content type, platform fit, source, personal
connection, technical depth and novelty.

The pool is deliberately oversampled (default twice the needed count) so the
weakest ideas can be dropped rather than written up. Generating 70 posts from
70 ideas means writing up 70 mediocre ideas.

### Evaluation

Six evaluators, each returning a score and a reason.

**Authenticity** asks the product question directly: would a reader believe
this person had this thought.

**Voice match** compares the draft against both voice layers.

**Slop detection** runs from a YAML rule file rather than a hardcoded
blacklist, so the phrase patterns, structural tells, enthusiasm markers and
formatting excesses can each be weighted and toggled. The model-based pass
reads the same file, so tuning one config changes both halves.

**Repetition** runs four detectors, because "have I said this before" has four
different meanings:

- *Exact*, by normalised hash and character trigram similarity
- *Topic*, by topic key with a cooldown window
- *Idea*, by a cheap semantic pass over near neighbours only, so the cost
  stays bounded as history grows
- *Structural*, by fingerprinting the shape of the post and capping how often
  one shape appears in a week
- *Emotional*, by classifying the opening move, so a week is not ten variants
  of "I realized"

**Technical accuracy** separates fact from inference from opinion from
simplification, and flags uncertainty rather than resolving it confidently.
This matters most for cryptography and ZK content, where a confident wrong
explanation is worse than no post.

**Monetization** is covered below.

Failing drafts are regenerated with the failure reason fed back in, bounded by
`max_regeneration_attempts`, then flagged for review rather than silently
shipped.

### Diagram engine

Every LinkedIn post gets a diagram, and X posts get one where it earns its
place. For content about proof systems this is not decoration: the whole
argument in a post about Spartan is a reduction chain, and a chain is a
picture. A reader who cannot follow four paragraphs of multilinear extensions
can follow four boxes and three arrows.

The split that keeps this reliable: **a model decides the content of the
diagram, code decides how it looks.** The generator emits a structured spec
(nodes, edges, kind, labels) rather than an image or drawing instructions.
A deterministic renderer turns that spec into a PNG using a fixed house style.

That split buys three things. Rendering is testable, since the same spec
always produces the same image. Every diagram looks like it came from the same
person, which matters when they accumulate on a profile. And a style change
regenerates every past image without asking a model to reinvent it, which is
why `Visual` stores the spec alongside the file path.

Diagram kinds, chosen to match what this account actually writes about:

| Kind | For |
|---|---|
| `chain` | Reductions: computation to R1CS to polynomial to one evaluation |
| `flow` | Protocols with branching, rounds, or a prover and verifier exchange |
| `comparison` | Two systems side by side, such as Groth16 against Spartan |
| `timeline` | Sequences of rounds, or an incident unfolding |
| `plot` | Actual data, when there is any |

### Scheduler

Assigns each approved draft a date and time across the coming week, spread
over the configured posting windows by weight, respecting minimum gaps, with a
random per-slot offset.

The jitter is not cosmetic. Machine-regular posting cadence is a signal the
ranking model uses against an account, so evenly spaced round-numbered times
are actively harmful.

### Excel export

The workbook is the weekly deliverable, not an export feature. Six sheets:
weekly calendar, unused ideas, research, voice feedback, content history, and
monetization tracking.

It is re-importable. Edits and status changes you make in the sheet flow back
into the database, which is what closes the learning loop for people who
would rather work in Excel than in a web app.

## X monetization as a constraint layer

The account is being built toward the X Original Content Rewards program, so
several program rules are encoded as constraints rather than preferences.

| Program rule | How it lands in the system |
|---|---|
| Engagement-bait calls to action are a violation | Hard fail, not a score penalty |
| Content must be original | Provenance tagged per post: primary work, or commentary with substantive analysis |
| Ranking rewards reply depth and dwell time | Reply-worthiness scored separately from bait |
| Machine-regular behaviour is demoted | Scheduler jitter, varied structure |
| Topic scattering resets interest-graph classification | Niche coherence checked across the batch |
| Eligibility: 500 verified followers, 500,000 verified impressions in 90 days | Tracked in the workbook against both gates |

Worth stating plainly: this does not turn the system into an engagement
optimiser. The program pays for original content with a real point of view,
which is the same thing the authenticity engine is already trying to produce.
Where they would diverge, the priority order in `CLAUDE.md` holds and
authenticity wins.

## Model access

`LLMProvider` is a narrow protocol. Three implementations:

| Provider | Auth | Notes |
|---|---|---|
| `AgentSDKProvider` | Claude subscription | No API key. The default. |
| `AnthropicProvider` | API key | Prompt caching and batching. Better at volume. |
| `MockProvider` | none | Deterministic, offline, free. Runs the whole pipeline in tests. |

Claude Pro and the Anthropic API are separate products with separate billing;
a subscription does not include API access. The Agent SDK runs on the
subscription, which is why it is the default here.

The request type carries a sequence of system blocks rather than one string.
The stable prefix of a prompt (persona, voice profile, slop rules) repeats
across roughly a hundred calls in a weekly run, so marking where it ends lets
the Messages API provider place a cache breakpoint. Providers that cannot
cache concatenate and ignore the flag. Nothing at a call site changes when you
switch.

Sampling parameters are deliberately absent from the request type. The target
models reject them, and steering happens through the prompt and the effort
level.

## Prompt architecture

Fragments live in separate files and are composed at runtime:
`BASE_PERSONA`, `VOICE_PROFILE`, `PLATFORM_X`, `PLATFORM_LINKEDIN`,
`CONTENT_TYPE`, `RESEARCH_CONTEXT`, `PERSONAL_CONTEXT`,
`AUTHENTICITY_EVALUATOR`, `SLOP_DETECTOR`, `TECHNICAL_FACT_CHECKER`,
`MONETIZATION_RULES`.

Ordering is not arbitrary: stable fragments come first so the cache
breakpoint sits as late as possible, and nothing volatile (a date, a run id)
appears in an early fragment.

## Storage

SQLite through SQLModel, with Alembic migrations from the first commit so
moving to Postgres is a connection string.

Core tables follow the brief. Three additions the pipeline needs:

- `evaluations`, one row per draft per evaluator, so scores are auditable and
  trends are visible rather than being a single opaque number
- `schedule_slots`, so the calendar can be reshuffled without regenerating
  content
- `monetization_snapshots`, tracking follower count and 90-day verified
  impressions against the program gates

## Testing

`MockProvider` makes an end to end weekly run testable offline and for free,
which is what makes the pipeline safe to refactor.

The regression corpus in `tests/fixtures/` holds labelled examples across
seven categories: good, bad, AI slop, authentic, too generic, too promotional
and too technical. Evaluators are asserted against it, so scoring cannot drift
quietly as prompts change. Adversarial fixtures specifically target the
experience invariant.

## Build order

Phases ship one per pull request. Research and the dashboard come after the
weekly run works, because neither is in the minimum path from writing samples
to a usable Excel workbook, and the workbook is the thing that makes the
system worth using.
