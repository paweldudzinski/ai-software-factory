import asyncio
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from software_factory.application.ports.llm_provider import LLMProviderError
from software_factory.bootstrap import build_ask_llm
from software_factory.domain.llm.message import LLMMessage, MessageRole
from software_factory.domain.llm.request import LLMRequest
from software_factory.domain.llm.response import LLMResponse, StopReason
from software_factory.infrastructure.config.loader import (
    ConfigError,
    load_config,
)
from software_factory.infrastructure.config.schema import FactoryConfig

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

    typer.echo(" ")
    console.print(f"[bold green]{TITLE}[/]")
    typer.echo("-" * 19)
    for label, value in rows:
        typer.echo(f"{label + ':':<{width}}{value}")


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="Question or instruction for the LLM.")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the factory config file."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Send a single prompt to the configured LLM and print its response."""
    try:
        factory_config = load_config(config)
    except ConfigError as exc:
        _fail(str(exc))

    try:
        request = LLMRequest(
            model=factory_config.llm.model,
            max_tokens=factory_config.llm.max_output_tokens,
            messages=[LLMMessage(role=MessageRole.USER, content=prompt)],
        )
    except ValidationError:
        _fail("Prompt must not be empty.")

    try:
        ask_llm = build_ask_llm(factory_config)
        with console.status("Waiting for the LLM..."):
            response = asyncio.run(ask_llm.execute(request))
    except LLMProviderError as exc:
        message = str(exc)
        if exc.request_id:
            message += f"\nRequest ID: {exc.request_id}"
        _fail(message)

    _print_response(factory_config, response)


def _print_response(factory_config: FactoryConfig, response: LLMResponse) -> None:
    separator = "-" * 19

    typer.echo()
    console.print(f"[bold green]{TITLE}[/]")
    typer.echo(separator)
    typer.echo()
    typer.echo(f"Provider: {factory_config.llm.provider.capitalize()}")
    typer.echo(f"Model: {response.model}")
    typer.echo()
    typer.echo("Response:")
    typer.echo(response.content or "(no text in response)")
    if response.stop_reason is not StopReason.END_TURN:
        console.print(
            f"[bold yellow]Warning:[/] response stopped early "
            f"(stop reason: {response.stop_reason.value})"
        )
    typer.echo()
    typer.echo(separator)
    typer.echo(f"Input tokens: {response.usage.input_tokens}")
    typer.echo(f"Output tokens: {response.usage.output_tokens}")
    typer.echo(f"Stop reason: {response.stop_reason.value}")
    typer.echo(f"Request ID: {response.request_id or 'n/a'}")


def _fail(message: str) -> NoReturn:
    console.print(f"[bold red]{escape(message)}[/]")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
