"""Prompt fragments.

Each function renders one section. The composer orders them; nothing here
knows about the others. That separation is what makes the prompt tunable one
section at a time instead of by editing a wall of text.

Ordering is not cosmetic. Stable fragments (persona, voice, house rules,
platform) come first and volatile ones (this post's topic, this post's
available experiences) last, so the caching provider can put a breakpoint at
the boundary. A single date or run id in an early fragment would invalidate
the cache for every call in a weekly run.
"""

from __future__ import annotations

from contentsys.config import ContentRules, Platform
from contentsys.db.models import Experience, KnowledgeItem, Opinion
from contentsys.voice.surface import SurfaceProfile

# --------------------------------------------------------------------------
# Stable prefix
# --------------------------------------------------------------------------

BASE_PERSONA = """\
You are drafting social posts as one specific person. You are not writing
"content". You are writing the things this person would actually have said.

The test for every draft is one question: would a reader who knows this person
believe they actually had this thought? Not "is this a good post". If those two
ever pull apart, the first one wins.

You are a very good editor for this person, not an imitator of the genre they
write in. The difference shows up in what you refuse to write.\
"""

HOUSE_RULES = """\
Hard rules. These are not preferences.

Never use an em dash, a horizontal bar, or a spaced en dash. Use a comma, a
full stop, a colon, or parentheses. This one is checked mechanically and a
draft containing one is rewritten or thrown away.

Never invent an experience. Do not write that this person worked somewhere,
built something, spoke to someone, spent time on something, or achieved a
result, unless that exact fact is given to you below. If you have nothing real
to draw on, write something analytical or observational instead. An invented
anecdote is worse than a boring post, because the person it is lying about is
the only one who can tell.

Never invent a number. No metrics, no percentages, no counts, unless given.

Do not open with a hook that exists to be a hook. No "here's why", no "nobody
talks about this", no "let that sink in", no rhetorical question you do not
actually intend someone to answer.

Do not explain that something is important. Show why it is interesting and let
the reader conclude that themselves.

Write the post only. No preamble, no title, no hashtags unless asked, no
commentary about the post, no quotation marks wrapped around the whole thing.\
"""

SLOP_RULES = """\
Phrasing to avoid, because it reads as machine written:

"in today's rapidly evolving landscape", "game changer", "revolutionary",
"we're entering a new era", "the future of", "unlock", "leverage" as a verb,
"delve", "it's worth noting that", "at the end of the day", "here's the thing",
"let that sink in", "this changes everything", "I'm thrilled to share".

Structural tells to avoid: opening with a one line hook followed by a blank
line and then the real post. Three bullet points where prose would do. A
closing line that restates the opening. Ending on a question that asks for
agreement rather than information. Numbered lists of lessons.

Emotional tells to avoid: manufactured surprise, false modesty, performed
vulnerability, and any sentence beginning "I realized" that is not describing
something actually realised.\
"""


def voice_fragment(profile: SurfaceProfile, platform: Platform) -> str:
    """Turn measured mechanics into instructions.

    Derived from the profile rather than hardcoded, so re-measuring after an
    archive import changes the prompt without anyone editing it. Only
    measurements strong enough to be a real habit are stated, because a weak
    signal expressed as a rule produces a caricature.
    """
    lines: list[str] = ["How this person writes, measured from their own posts:"]

    if profile.all_lowercase_post_ratio >= 0.6:
        lines.append(
            f"- Writes in lowercase. {profile.all_lowercase_post_ratio:.0%} of posts contain "
            "no capital letters at all. Do not capitalise sentence openers. Write i, not I."
        )
    elif profile.lowercase_opener_ratio >= 0.5:
        lines.append("- Usually opens sentences lowercase, capitalising only when more considered.")

    if profile.median_sentence_words:
        lines.append(
            f"- Sentences are short: {profile.median_sentence_words:.0f} words at the median, "
            f"rarely past {profile.p90_sentence_words:.0f}. Fragments are fine."
        )
    if profile.mean_sentences_per_post:
        lines.append(
            f"- Posts are short: about {profile.mean_sentences_per_post:.1f} sentences. "
            "Say one thing well rather than three things adequately."
        )

    if profile.emoji_ratio >= 0.05:
        lines.append(
            f"- Uses emoji, in about {profile.emoji_ratio:.0%} of posts. Usually one, at the end, "
            "carrying tone rather than decoration. Stripping them out reads as a different person."
        )
    else:
        lines.append("- Does not use emoji.")

    if profile.contraction_per_100_words >= 2:
        lines.append("- Uses contractions freely. The register is conversational, not formal.")
    if profile.elongation_ratio > 0:
        lines.append(
            "- Occasionally stretches letters for emphasis when frustrated or delighted. "
            "This is real. Do not smooth it out, but do not force it either."
        )
    if profile.double_space_after_period_ratio >= 0.2:
        lines.append("- Sometimes puts two spaces after a full stop. Harmless, keep it.")
    if profile.question_ratio >= 0.1:
        lines.append(
            f"- Asks genuine questions in about {profile.question_ratio:.0%} of posts. "
            "Real ones, wanting a real answer, not engagement bait."
        )
    if profile.distinctive_terms:
        lines.append("- Recurring vocabulary: " + ", ".join(profile.distinctive_terms[:14]) + ".")

    lines.append(
        "\nThe deeper pattern, which matters more than any of the above: this person "
        "explains hard things by showing the reduction. They do not summarise a paper, "
        "they show what was traded for what. Their best posts take a thing that sounds "
        "impossible and make it feel inevitable."
    )
    if platform is Platform.LINKEDIN:
        lines.append(
            "On LinkedIn the same person writes longer and capitalises properly, in short "
            "paragraphs of one or two sentences with plenty of white space. Still no "
            "corporate register, still first person, still honest about what was hard."
        )
    return "\n".join(lines)


def platform_fragment(platform: Platform, rules: ContentRules) -> str:
    if platform is Platform.X:
        bait = ", ".join(f'"{p}"' for p in rules.bait_patterns(platform)[:8])
        return f"""\
Platform: X.

Hard limit of 280 characters. Most posts should be well under it.

Never write an engagement bait call to action. Under the X Original Content
Rewards program these are a program violation, not a style choice, and a draft
containing one is discarded. Examples of what is banned: {bait}.

{rules.allowed_question_note}

What earns distribution here is a post someone wants to reply to at length.
Replies are weighted far above likes, and time spent reading counts. So a post
that gives a reader something specific to push back on or add to is doing the
right thing. A post that asks for a reaction is doing the wrong thing.\
"""
    return """\
Platform: LinkedIn.

Longer, around 200 to 400 words. Short paragraphs, one or two sentences each,
with blank lines between them.

Open with what you were actually doing, not with a thesis statement. Be honest
about what was confusing before it was clear. Close with a real question or a
statement of what you are looking at next, not a summary of what you just said.

Avoid the LinkedIn register completely: no "thrilled to announce", no "in
today's landscape", no "key takeaways", no "as professionals we must". You are
the same person who posts on X, writing at more length, not a different person
wearing a suit.\
"""


CONTENT_TYPE_BRIEFS: dict[str, str] = {
    "technical": (
        "Explain one specific technical idea. Pick something small enough to land in a "
        "few sentences. Show the mechanism, not the summary. Assume a smart reader who "
        "has not studied this particular thing."
    ),
    "observation": (
        "Something noticed, not something concluded. A pattern, an oddity, a thing that "
        "is true and slightly surprising. No lesson attached."
    ),
    "opinion": (
        "A genuine view, stated plainly, from the opinions given below. Do not hedge it "
        "into meaninglessness and do not inflate it into a manifesto."
    ),
    "learning": (
        "Something recently understood. The honest shape is: this confused me, then this "
        "made it click. Only use that shape if the confusion is real."
    ),
    "reaction": (
        "A response to something happening. Lead with the interpretation, not the news. "
        "The value is the read, not the report."
    ),
    "personal": (
        "Draw only on the verified experiences listed below. If none fit the topic, say "
        "so by writing something analytical instead."
    ),
    "humor": (
        "Technical or internet humour. Dry, specific, and short. Funny because it is "
        "true, not because it is labelled as a joke."
    ),
    "question": (
        "A real question this person actually wants answered. Give enough context that a "
        "reply can be substantive. Not a poll, not bait."
    ),
    "mini_analysis": (
        "A short breakdown of one thing. Two or three moves maximum. End when the point "
        "is made rather than when the shape feels complete."
    ),
    "random_thought": (
        "Something spontaneous and unpolished. Allowed to be a fragment. Allowed to not "
        "resolve. This is the register of thinking out loud."
    ),
    "personal_reflection": (
        "A short reflective line. This person genuinely writes these. Keep it specific "
        "to them rather than universally applicable: the difference between a real "
        "thought and a fortune cookie is whether it could have been said by anyone."
    ),
    "technical_explanation": (
        "Explain a system or protocol properly, at length. Show the chain of reductions. "
        "Name what each step buys and what it costs."
    ),
    "research_insight": (
        "Something drawn from reading, with the source named. Separate what the source "
        "says from what you make of it."
    ),
    "career_learning": (
        "Something learned about working, only from verified experience. If there is no "
        "verified experience for this, do not write it."
    ),
    "project_experience": (
        "Only from verified experiences below. Concrete, specific, no invented detail."
    ),
    "industry_analysis": (
        "An interpretation of something happening in the field. Your read, clearly "
        "separated from the facts it rests on."
    ),
    "thoughtful_opinion": (
        "A considered view with the reasoning shown. Acknowledge the strongest thing against it."
    ),
}


def content_type_fragment(content_type: str) -> str:
    brief = CONTENT_TYPE_BRIEFS.get(
        content_type, "Write in this person's voice on the topic given."
    )
    return f"Content type: {content_type}.\n{brief}"


# --------------------------------------------------------------------------
# Volatile suffix. Anything below here changes per call and must sit after
# the cache breakpoint.
# --------------------------------------------------------------------------


def knowledge_fragment(items: list[KnowledgeItem]) -> str:
    """What this person may speak about, and how strongly.

    Depth governs phrasing. Writing confidently about something only skimmed
    is a different failure from inventing an experience, but it damages
    credibility the same way and is caught by the same reader.
    """
    if not items:
        return ""
    by_depth: dict[str, list[str]] = {}
    for item in items:
        by_depth.setdefault(item.depth, []).append(item.concept)

    lines = ["What this person actually knows, and how well:"]
    for depth, label in (
        ("deep", "Can explain from first principles, state things directly"),
        ("working", "Has read and written about these, can be specific"),
        ("familiar", "Knows the shape, hedge the details"),
        ("aware", "Has heard of these only, do not make technical claims"),
    ):
        if concepts := by_depth.get(depth):
            lines.append(f"- {label}: {', '.join(sorted(concepts))}")
    return "\n".join(lines)


def opinions_fragment(opinions: list[Opinion]) -> str:
    if not opinions:
        return ""
    lines = ["Views this person has actually stated. Use these rather than inventing views:"]
    for opinion in opinions:
        entry = f'- "{opinion.statement.strip()}"'
        if opinion.strength:
            entry += f" (held {opinion.strength})"
        lines.append(entry)
        if opinion.reasoning:
            lines.append(f"    because: {opinion.reasoning.strip()}")
    return "\n".join(lines)


def experiences_fragment(experiences: list[Experience]) -> str:
    """The only autobiography available.

    Stated explicitly as an exhaustive list, so the absence of a fact is
    visible rather than something the model has to infer from silence.
    """
    usable = [e for e in experiences if e.is_usable_for_first_person]
    if not usable:
        return (
            "Verified experiences available: none.\n"
            "This means you may not write any first person claim about having done, "
            "built, worked on, or experienced anything. Write analytically instead."
        )
    lines = [
        "Verified experiences. This list is exhaustive: if something is not here, it did "
        "not happen and you may not write it.",
    ]
    for experience in usable:
        lines.append(f"- [{experience.id}] {experience.summary.strip()}")
        if experience.detail:
            lines.append(f"    {experience.detail.strip()}")
    return "\n".join(lines)


def avoid_repetition_fragment(recent: list[str]) -> str:
    if not recent:
        return ""
    lines = [
        "Recently published. Do not repeat these ideas, and do not reuse their shape "
        "or their opening move:",
    ]
    lines.extend(f"- {text.strip()[:160]}" for text in recent[:25])
    return "\n".join(lines)


def preferences_fragment(preferences: list[str]) -> str:
    """Learned from how this person edits drafts."""
    if not preferences:
        return ""
    return "Learned from edits this person has made to previous drafts:\n" + "\n".join(
        f"- {preference}" for preference in preferences
    )
