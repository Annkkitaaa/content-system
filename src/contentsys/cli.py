"""Command line entrypoint.

Commands are added as their phases land. Right now this exposes enough to
confirm the install works and to inspect the resolved configuration, which is
the first thing worth checking when something behaves unexpectedly.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from contentsys import __version__
from contentsys.config import Platform, get_settings

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


if __name__ == "__main__":
    app()
