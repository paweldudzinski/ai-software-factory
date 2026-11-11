from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from software_factory.infrastructure.config.loader import ConfigError, load_config

DEFAULT_CONFIG_PATH = Path("config/factory.yaml")
TITLE = "AI Software Factory"

app = typer.Typer(name="factory", help="Software Factory CLI.", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """Software Factory CLI."""


def _package_version() -> str:
    try:
        return version("software-factory")
    except PackageNotFoundError:
        return "unknown"


@app.command()
def info(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the factory config file."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Show the factory status."""
    try:
        factory_config = load_config(config)
    except ConfigError as exc:
        console.print(f"[bold red]{escape(str(exc))}[/]")
        raise typer.Exit(code=1)

    # TODO: replace with real counts once agents and the task queue exist.
    active_agents = 0
    active_tasks = 0

    rows = [
        ("Version", _package_version()),
        ("Environment", factory_config.factory.environment),
        ("Provider", factory_config.llm.provider),
        ("Runtime", "ready"),
        ("Active agents", active_agents),
        ("Active tasks", active_tasks),
    ]
    width = max(len(label) for label, _ in rows) + 2

    typer.echo(TITLE)
    typer.echo("-" * 19)
    for label, value in rows:
        typer.echo(f"{label + ':':<{width}}{value}")


if __name__ == "__main__":
    app()
