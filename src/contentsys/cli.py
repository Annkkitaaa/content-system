"""Command line entrypoint.

Commands are added as their phases land. Right now this exposes enough to
confirm the install works and to inspect the resolved configuration, which is
the first thing worth checking when something behaves unexpectedly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlmodel import func, select

from contentsys import __version__
from contentsys.config import Platform, get_settings
from contentsys.db import create_all, session_scope
from contentsys.db.models import (
    Experience,
    KnowledgeItem,
    Opinion,
    SampleSource,
    Topic,
    WritingSample,
)
from contentsys.knowledge import import_samples, load_seed
from contentsys.voice import (
    MIN_USEFUL_SAMPLES,
    active_profile,
    build_profile,
    load_surface,
    samples_for,
)

app = typer.Typer(
    name="contentsys",
    help="Turn your own knowledge and voice into authentic X and LinkedIn drafts.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"contentsys {__version__}")


@app.command("config")
def show_config() -> None:
    """Show the resolved configuration.

    Reads .env and the YAML files under config/, then prints what the
    pipeline will actually use. Secrets are reported as present or absent,
    never echoed.
    """
    settings = get_settings()

    general = Table(title="General", show_header=False, box=None, pad_edge=False)
    general.add_column(style="cyan")
    general.add_column()
    general.add_row("provider", settings.provider.value)
    general.add_row(
        "generation model", f"{settings.generation_model} (effort {settings.generation_effort})"
    )
    general.add_row(
        "evaluation model", f"{settings.evaluation_model} (effort {settings.evaluation_effort})"
    )
    general.add_row("timezone", settings.timezone)
    general.add_row("database", settings.database_url)
    general.add_row("exports", str(settings.export_dir))
    general.add_row("anthropic api key", "set" if settings.anthropic_api_key else "not set")
    console.print(general)

    thresholds = Table(title="Regeneration thresholds", show_header=False, box=None, pad_edge=False)
    thresholds.add_column(style="cyan")
    thresholds.add_column()
    thresholds.add_row("authenticity", f">= {settings.thresholds.authenticity}")
    thresholds.add_row("originality", f">= {settings.thresholds.originality}")
    thresholds.add_row("voice match", f">= {settings.thresholds.voice_match}")
    thresholds.add_row("technical accuracy", f">= {settings.thresholds.technical_accuracy}")
    thresholds.add_row("slop risk", f"at most {settings.thresholds.max_slop_risk.value}")
    thresholds.add_row(
        "repetition risk", f"at most {settings.thresholds.max_repetition_risk.value}"
    )
    thresholds.add_row("max regenerations", str(settings.thresholds.max_regeneration_attempts))
    console.print(thresholds)

    schedule = settings.schedule
    weekly = Table(title=f"Weekly plan ({schedule.weekly_total()} pieces)")
    weekly.add_column("Platform")
    weekly.add_column("Per week", justify="right")
    weekly.add_column("Content types")
    for platform in Platform:
        plan = schedule.for_platform(platform)
        allocation = settings.content_mix.allocate(platform, plan.weekly_total())
        summary = ", ".join(
            f"{name} {count}" for name, count in sorted(allocation.items()) if count
        )
        weekly.add_row(platform.value, str(plan.weekly_total()), summary)
    console.print(weekly)


@app.command("init")
def init_db() -> None:
    """Create the database and its tables."""
    create_all()
    console.print(f"[green]Ready.[/green] Database at {get_settings().database_url}")


@app.command("seed")
def seed() -> None:
    """Load the starting knowledge base from seed/.

    Safe to run more than once. Anything already present is left alone rather
    than duplicated, because duplicates quietly bias the voice measurement.
    """
    create_all()
    with session_scope() as session:
        reports = load_seed(session)

    table = Table(title="Seed loaded")
    table.add_column("Kind")
    table.add_column("Result")
    for kind, report in reports.items():
        table.add_row(kind, report.describe())
    console.print(table)
    console.print("\nNext: [cyan]contentsys voice build[/cyan]")


@app.command("import-samples")
def import_samples_command(
    path: Annotated[Path, typer.Argument(help="A .txt, .md, .csv or .json export")],
    source: Annotated[
        SampleSource, typer.Option("--source", "-s", help="Where the writing came from")
    ] = SampleSource.X,
) -> None:
    """Import your own writing.

    Everything here is ground truth for the voice engine, so only your words
    belong in it. Duplicates are detected by content, which means re-importing
    an overlapping export is harmless.
    """
    if not path.exists():
        console.print(f"[red]No such file:[/red] {path}")
        raise typer.Exit(1)

    create_all()
    with session_scope() as session:
        report = import_samples(session, path, source)

    console.print(f"[green]{source.value}:[/green] {report.describe()}")
    if report.added:
        console.print("\nNext: [cyan]contentsys voice build[/cyan]")


from contentsys.cli_generate import app as generate_app  # noqa: E402

app.add_typer(generate_app, name="generate")

voice_app = typer.Typer(help="Inspect and rebuild the voice profile.", no_args_is_help=True)
app.add_typer(voice_app, name="voice")


@voice_app.command("build")
def voice_build() -> None:
    """Measure your writing and store a new voice profile."""
    create_all()
    with session_scope() as session:
        for platform in Platform:
            profile, surface = build_profile(session, platform)
            header = f"{platform.value}: version {profile.version}, {profile.sample_count} samples"
            if profile.sample_count == 0:
                console.print(f"[yellow]{header}. Nothing to measure yet.[/yellow]")
                continue
            colour = "green" if profile.sample_count >= MIN_USEFUL_SAMPLES else "yellow"
            console.print(f"[{colour}]{header}[/{colour}]")
            for line in surface.describe():
                console.print(f"  {line}")
            if profile.sample_count < MIN_USEFUL_SAMPLES:
                console.print(
                    f"  [yellow]Under {MIN_USEFUL_SAMPLES} samples, so treat these numbers as "
                    "provisional. More writing sharpens them fast.[/yellow]"
                )
            console.print()


@voice_app.command("show")
def voice_show(
    platform: Annotated[Platform, typer.Argument(help="X or LinkedIn")] = Platform.X,
) -> None:
    """Show the stored voice profile for a platform."""
    with session_scope() as session:
        profile = active_profile(session, platform)
        if profile is None:
            console.print("[yellow]No profile yet. Run 'contentsys voice build'.[/yellow]")
            raise typer.Exit(1)
        surface = load_surface(profile)
        count = profile.sample_count
        version = profile.version

    console.print(f"[bold]{platform.value}[/bold]  version {version}, {count} samples\n")
    for line in surface.describe():
        console.print(f"  {line}")

    table = Table(title="\nMeasurements", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in surface.to_dict().items():
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value[:10]) or "none"
        table.add_row(key.replace("_", " "), str(value))
    console.print(table)


@app.command("weekly")
def weekly(
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Override the configured provider")
    ] = None,
    x_posts: Annotated[int | None, typer.Option("--x-posts", help="X posts per day")] = None,
    linkedin_posts: Annotated[
        int | None, typer.Option("--linkedin-posts", help="LinkedIn posts this week")
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Make the calendar reproducible")
    ] = None,
    research: Annotated[
        bool,
        typer.Option(
            "--research/--no-research",
            help="Pull recent news so part of the week reacts to it",
        ),
    ] = True,
) -> None:
    """Generate the coming week and write the workbook.

    Nothing is published. The workbook is a plan you edit and work from.
    """
    from contentsys.llm.registry import build_provider
    from contentsys.pipeline import run_weekly

    settings = get_settings()
    create_all()
    llm = build_provider(provider, settings)

    console.print(
        f"[dim]Provider {llm.name}, model {settings.generation_model}. "
        "Nothing will be published.[/dim]\n"
    )

    with session_scope() as session:
        result = run_weekly(
            session,
            llm,
            settings,
            x_posts=x_posts,
            linkedin_posts=linkedin_posts,
            seed=seed,
            use_research=research,
            progress=lambda message: console.print(f"[dim]{message}[/dim]"),
        )

    console.print()
    for line in result.summary:
        console.print(line)

    usage = result.usage
    console.print(
        f"\n[dim]{usage.input_tokens + usage.output_tokens} tokens billed, "
        f"{usage.cache_read_tokens} served from cache[/dim]"
    )
    console.print(f"\n[green]Written to[/green] {result.path}")
    console.print(
        "[dim]Edit it, set Status as you approve, then feed edits back with "
        "'contentsys teach'.[/dim]"
    )


@app.command("monetization")
def monetization(
    followers: Annotated[int | None, typer.Option("--followers")] = None,
    impressions: Annotated[
        int | None, typer.Option("--impressions", help="Verified impressions over 90 days")
    ] = None,
    premium: Annotated[bool | None, typer.Option("--premium/--no-premium")] = None,
) -> None:
    """Record where the account stands against the program gates."""
    from contentsys.pipeline import record_snapshot

    create_all()
    with session_scope() as session:
        snapshot = record_snapshot(
            session,
            verified_followers=followers,
            verified_impressions_90d=impressions,
            premium_active=premium,
        )
        captured = snapshot.captured_on

    console.print(f"[green]Recorded for {captured}.[/green] It appears in the next workbook.")


@app.command("teach")
def teach(
    draft: Annotated[Path, typer.Argument(help="File holding the draft as generated")],
    edited: Annotated[Path, typer.Argument(help="File holding your rewritten version")],
) -> None:
    """Learn from how you rewrote a draft.

    A preference is recorded the first time it is seen but not used. It only
    reaches the prompt once the same change shows up repeatedly, so one
    unusual edit cannot permanently reshape how everything is written.
    """
    from contentsys.voice import ACTIVE_CONFIDENCE, active_preferences, learn_from

    for path in (draft, edited):
        if not path.exists():
            console.print(f"[red]No such file:[/red] {path}")
            raise typer.Exit(1)

    original = draft.read_text(encoding="utf-8")
    revised = edited.read_text(encoding="utf-8")

    create_all()
    with session_scope() as session:
        report = learn_from(session, original, revised)
        session.flush()
        preferences = [(p.key, p.description, p.confidence) for p in active_preferences(session)]

    console.print(f"[bold]What changed:[/bold] {report.analysis.summary()}")
    if report.analysis.unclassified:
        console.print(
            "[yellow]The text changed but nothing recognisable did, so nothing was "
            "learned. This is deliberate: a wrong lesson learned confidently is "
            "worse than no lesson.[/yellow]"
        )
    console.print(f"[bold]Result:[/bold] {report.describe()}")

    if preferences:
        table = Table(title=f"\nIn use (confidence >= {ACTIVE_CONFIDENCE})")
        table.add_column("Preference")
        table.add_column("Seen", justify="right")
        for _, description, confidence in preferences:
            table.add_row(description, str(confidence))
        console.print(table)
    else:
        console.print(
            f"\n[dim]Nothing is confident enough to use yet. A preference needs "
            f"{ACTIVE_CONFIDENCE} observations before it reaches a prompt.[/dim]"
        )


@app.command("forget")
def forget_preference(
    key: Annotated[str, typer.Argument(help="The preference key to drop")],
) -> None:
    """Drop a learned preference.

    The system will occasionally learn something wrong, and a voice model with
    no undo is one nobody will trust enough to keep feeding.
    """
    from contentsys.voice import forget

    with session_scope() as session:
        removed = forget(session, key)

    if removed:
        console.print(f"[green]Forgot[/green] {key}")
    else:
        console.print(f"[yellow]No preference called[/yellow] {key}")


@app.command("knowledge")
def knowledge_summary() -> None:
    """Show what the system knows about you.

    Worth checking before the first generation run: content quality is capped
    by what is in here, and an empty knowledge base produces generic posts no
    matter how good the prompts are.
    """
    with session_scope() as session:
        counts = {
            "writing samples": session.exec(select(func.count()).select_from(WritingSample)).one(),
            "experiences": session.exec(select(func.count()).select_from(Experience)).one(),
            "opinions": session.exec(select(func.count()).select_from(Opinion)).one(),
            "knowledge items": session.exec(select(func.count()).select_from(KnowledgeItem)).one(),
            "topics": session.exec(select(func.count()).select_from(Topic)).one(),
        }
        usable = sum(
            1
            for experience in session.exec(select(Experience))
            if experience.is_usable_for_first_person
        )
        deep = list(
            session.exec(
                select(KnowledgeItem).where(KnowledgeItem.depth.in_(["deep", "working"]))  # type: ignore[attr-defined]
            )
        )
        per_platform = {platform: len(samples_for(session, platform)) for platform in Platform}
        deep_concepts = [item.concept for item in deep]

    table = Table(title="Knowledge base")
    table.add_column("Kind")
    table.add_column("Count", justify="right")
    for label, value in counts.items():
        table.add_row(label, str(value))
    console.print(table)

    console.print(
        f"\n[bold]{usable}[/bold] experiences are confirmed and usable for first person claims."
    )
    if usable < counts["experiences"]:
        console.print(
            f"[yellow]{counts['experiences'] - usable} are unconfirmed and cannot back a "
            "personal post until you confirm them.[/yellow]"
        )

    console.print("\nUsable samples per platform:")
    for platform, count in per_platform.items():
        marker = "green" if count >= MIN_USEFUL_SAMPLES else "yellow"
        console.print(f"  [{marker}]{platform.value}: {count}[/{marker}]")

    if deep_concepts:
        console.print(
            f"\nCan write with authority on {len(deep_concepts)} concepts: "
            + ", ".join(sorted(deep_concepts)[:12])
            + ("..." if len(deep_concepts) > 12 else "")
        )


if __name__ == "__main__":
    app()
