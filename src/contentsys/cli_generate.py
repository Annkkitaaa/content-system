"""Generation commands.

Kept in their own module because the root CLI is already the place everything
converges, and generation has enough surface to be worth reading on its own.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from contentsys.config import Platform, ProviderName, get_settings
from contentsys.content import modes
from contentsys.content.context import build_context
from contentsys.content.generate import Draft
from contentsys.db import create_all, session_scope
from contentsys.llm.registry import build_provider

console = Console()
app = typer.Typer(help="Generate drafts. Nothing is ever published.", no_args_is_help=True)

ProviderOption = Annotated[
    ProviderName | None,
    typer.Option("--provider", "-p", help="Override the configured provider"),
]
PlatformOption = Annotated[Platform, typer.Option("--platform", help="X or LinkedIn")]


def _show(drafts: list[Draft]) -> None:
    """Print drafts so they can actually be judged.

    Shows the issues alongside the text rather than hiding them behind a
    score. A draft flagged for review is still shown: the owner is the editor,
    and hiding weak output makes it harder to tell the system is struggling.
    """
    if not drafts:
        console.print("[yellow]Nothing generated.[/yellow]")
        return

    for index, draft in enumerate(drafts, start=1):
        if draft.needs_review:
            border, label = "red", "NEEDS REVIEW"
        elif draft.issues:
            border, label = "yellow", "has issues"
        else:
            border, label = "green", "clean"

        heading = f"{index}. {draft.content_type} on {draft.topic}"
        body = draft.content or "[no content produced]"
        if draft.repairs:
            body += f"\n\n[dim]repaired: {', '.join(draft.repairs)}[/dim]"
        if draft.issues:
            body += "\n\n[yellow]" + "\n".join(f"! {issue}" for issue in draft.issues) + "[/yellow]"
        if draft.attempts > 1:
            body += f"\n[dim]{draft.attempts} attempts[/dim]"

        console.print(Panel(body, title=heading, subtitle=label, border_style=border))

    clean = sum(1 for draft in drafts if draft.ok)
    flagged = sum(1 for draft in drafts if draft.needs_review)
    chars = [len(d.content) for d in drafts if d.content]

    summary = Table(show_header=False, box=None)
    summary.add_column(style="cyan")
    summary.add_column()
    summary.add_row("generated", str(len(drafts)))
    summary.add_row("clean", f"{clean} of {len(drafts)}")
    if flagged:
        summary.add_row("needs review", str(flagged))
    if chars:
        summary.add_row("length", f"{min(chars)} to {max(chars)} characters")
    total = sum((d.usage.input_tokens + d.usage.output_tokens) for d in drafts)
    cached = sum(d.usage.cache_read_tokens for d in drafts)
    summary.add_row("tokens", f"{total} billed, {cached} served from cache")
    console.print(summary)


def _prepare(provider_name: ProviderName | None, platform: Platform, content_type: str):
    settings = get_settings()
    create_all()
    provider = build_provider(provider_name, settings)
    return settings, provider, platform, content_type


@app.command("daily")
def daily(
    provider: ProviderOption = None,
    posts: Annotated[int, typer.Option("--posts", "-n", help="How many")] = 10,
) -> None:
    """Mode 1. A day of X posts across the content mix."""
    settings, llm, platform, _ = _prepare(provider, Platform.X, "technical")
    with session_scope() as session:
        context = build_context(session, platform, settings=settings)
        drafts = modes.daily_x(llm, context, settings, posts=posts)
    _show(drafts)


@app.command("linkedin")
def linkedin(
    provider: ProviderOption = None,
    posts: Annotated[int, typer.Option("--posts", "-n")] = 2,
) -> None:
    """Mode 2. This week's LinkedIn candidates."""
    settings, llm, platform, _ = _prepare(provider, Platform.LINKEDIN, "technical_explanation")
    with session_scope() as session:
        context = build_context(session, platform, settings=settings)
        drafts = modes.weekly_linkedin(llm, context, settings, posts=posts)
    _show(drafts)


@app.command("ideas")
def ideas(
    topic: Annotated[str, typer.Argument(help="What to generate ideas about")],
    provider: ProviderOption = None,
    platform: PlatformOption = Platform.X,
    count: Annotated[int, typer.Option("--count", "-n")] = 8,
) -> None:
    """Mode 3. Angles on a topic, without drafting them."""
    settings, llm, platform, _ = _prepare(provider, platform, "technical")
    with session_scope() as session:
        context = build_context(session, platform, settings=settings, topic=topic)
        pool = modes.from_topic(llm, context, settings, topic=topic, count=count)

    table = Table(title=f"Ideas: {topic}")
    table.add_column("Angle", max_width=60)
    table.add_column("Type")
    table.add_column("Novelty", justify="right")
    for idea in pool.usable:
        table.add_row(idea.angle, idea.content_type, f"{idea.novelty:.1f}")
    console.print(table)
    console.print(f"[dim]{pool.summary()}[/dim]")


@app.command("explain")
def explain_concept(
    concept: Annotated[str, typer.Argument(help="The concept to explain")],
    provider: ProviderOption = None,
    platform: PlatformOption = Platform.X,
) -> None:
    """Mode 6. Make a technical concept understandable without breaking it."""
    settings, llm, platform, _ = _prepare(provider, platform, "technical")
    with session_scope() as session:
        context = build_context(session, platform, settings=settings, topic=concept)
        draft = modes.explain(llm, context, settings, concept=concept)
    _show([draft])


@app.command("react")
def react(
    event: Annotated[str, typer.Argument(help="What happened")],
    provider: ProviderOption = None,
    platform: PlatformOption = Platform.X,
) -> None:
    """Mode 5. Your read on something, not a report of it."""
    settings, llm, platform, _ = _prepare(provider, platform, "reaction")
    with session_scope() as session:
        context = build_context(session, platform, settings=settings)
        draft = modes.reaction(llm, context, settings, event=event)
    _show([draft])


@app.command("personal")
def personal(
    provider: ProviderOption = None,
    platform: PlatformOption = Platform.X,
    experience_id: Annotated[int | None, typer.Option("--experience-id")] = None,
) -> None:
    """Mode 7. A post from a verified experience, or nothing."""
    settings, llm, platform, _ = _prepare(provider, platform, "personal")
    with session_scope() as session:
        context = build_context(session, platform, settings=settings)
        draft = modes.personal(llm, context, settings, experience_id=experience_id)
    _show([draft])


@app.command("diagram")
def diagram(
    text: Annotated[str, typer.Argument(help="The post to draw a diagram for")],
    provider: ProviderOption = None,
    platform: PlatformOption = Platform.X,
    name: Annotated[str, typer.Option("--name", help="Filename stem")] = "diagram",
) -> None:
    """Draw a diagram for a post.

    A model decides what the diagram says; deterministic code decides how it
    looks. If the post has no structure worth drawing, that is a real answer
    and nothing is written.
    """
    from contentsys.visuals import diagram_path, generate_diagram, render

    settings, llm, platform, _ = _prepare(provider, platform, "technical")
    with session_scope() as session:
        context = build_context(session, platform, settings=settings)
        spec = generate_diagram(
            llm,
            context,
            content=text,
            model=settings.generation_model,
            effort=settings.generation_effort,
        )

    if spec is None:
        console.print(
            "[yellow]No diagram. This post does not carry a structure worth drawing, "
            "or the model did not return a usable one.[/yellow]"
        )
        raise typer.Exit(0)

    path = render(spec, diagram_path(settings.export_dir, platform, name))
    console.print(f"[green]Wrote[/green] {path}")
    console.print(f"[dim]alt text: {spec.describe()}[/dim]")


@app.command("dump")
def dump(
    text: Annotated[
        str | None, typer.Argument(help="Your messy thought. Omit to read stdin.")
    ] = None,
    provider: ProviderOption = None,
    platform: PlatformOption = Platform.X,
) -> None:
    """Mode 8. Your own thought, cleaned up just enough to post.

    Reads stdin when no argument is given, so you can pipe a file or paste a
    few paragraphs without fighting shell quoting.
    """
    raw = text if text is not None else sys.stdin.read()
    if not raw.strip():
        console.print("[red]Nothing to work with.[/red]")
        raise typer.Exit(1)

    settings, llm, platform, _ = _prepare(provider, platform, "random_thought")
    with session_scope() as session:
        context = build_context(session, platform, settings=settings)
        draft = modes.brain_dump(llm, context, settings, text=raw)

    console.print(Panel(raw.strip(), title="what you wrote", border_style="dim"))
    _show([draft])
