from importlib.metadata import version
from pathlib import Path

import pytest
from typer.testing import CliRunner

from software_factory.cli.app import app

SAMPLE_CONFIG = """\
factory:
  name: test-factory
  environment: local

llm:
  provider: anthropic
  model: test-model

runtime:
  workspace: ./workspace
  max_iterations: 3
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "factory.yaml"
    path.write_text(SAMPLE_CONFIG)
    return path


def test_info_prints_status(runner: CliRunner, config_file: Path) -> None:
    result = runner.invoke(app, ["info", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert result.output == (
        " \n"
        "AI Software Factory\n"
        "-------------------\n"
        f"Version:       {version('software-factory')}\n"
        "Environment:   local\n"
        "Provider:      anthropic\n"
        "Runtime:       ready\n"
        "Active agents: 0\n"
        "Active tasks:  0\n"
    )


def test_info_uses_default_config_path(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "factory.yaml").write_text(SAMPLE_CONFIG)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0, result.output
    assert "Environment:   local" in result.output


def test_info_fails_when_config_missing(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["info", "--config", str(tmp_path / "missing.yaml")])

    assert result.exit_code == 1
    assert "Config file not found" in result.output


def test_info_fails_on_invalid_yaml(runner: CliRunner, tmp_path: Path) -> None:
    config = tmp_path / "factory.yaml"
    config.write_text("factory: [unclosed\n")

    result = runner.invoke(app, ["info", "--config", str(config)])

    assert result.exit_code == 1
    assert "Invalid YAML" in result.output


def test_no_args_shows_help(runner: CliRunner) -> None:
    result = runner.invoke(app, [])

    assert "info" in result.output


def test_info_fails_on_invalid_config(runner: CliRunner, tmp_path: Path) -> None:
    config = tmp_path / "factory.yaml"
    config.write_text(
        SAMPLE_CONFIG.replace("model: test-model", 'model: ""').replace(
            "max_iterations: 3", "max_iterations: -10"
        )
    )

    result = runner.invoke(app, ["info", "--config", str(config)])

    assert result.exit_code == 1
    assert "llm.model" in result.output
    assert "runtime.max_iterations" in result.output
