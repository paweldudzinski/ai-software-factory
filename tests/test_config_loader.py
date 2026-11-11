from pathlib import Path
from typing import Any

import pytest
import yaml

from software_factory.infrastructure.config.loader import ConfigError, load_config
from software_factory.infrastructure.config.schema import FactoryConfig

VALID_CONFIG: dict[str, Any] = {
    "factory": {"name": "software-factory", "environment": "local"},
    "llm": {"provider": "anthropic", "model": "claude-opus-5-5"},
    "runtime": {"workspace": "./workspace", "max_iterations": 10},
}


def _write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "factory.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _with(section: str, field: str, value: Any) -> dict[str, Any]:
    data = {name: dict(values) for name, values in VALID_CONFIG.items()}
    data[section][field] = value
    return data


def test_loads_valid_config(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, VALID_CONFIG))

    assert isinstance(config, FactoryConfig)
    assert config.factory.name == "software-factory"
    assert config.factory.environment == "local"
    assert config.llm.provider == "anthropic"
    assert config.llm.model == "claude-opus-5-5"
    assert config.runtime.workspace == Path("./workspace")
    assert config.runtime.max_iterations == 10


def test_reports_all_errors_from_example(tmp_path: Path) -> None:
    data = _with("llm", "model", "")
    data["runtime"]["max_iterations"] = -10

    with pytest.raises(ConfigError) as exc_info:
        load_config(_write(tmp_path, data))

    message = str(exc_info.value)
    assert "2 error(s)" in message
    assert "llm.model" in message
    assert "runtime.max_iterations" in message


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("factory", "name", ""),
        ("factory", "name", "   "),
        ("factory", "environment", "production"),
        ("llm", "provider", "openai"),
        ("llm", "model", ""),
        ("llm", "model", "   "),
        ("runtime", "max_iterations", 0),
        ("runtime", "max_iterations", -10),
        ("runtime", "max_iterations", "10"),
        ("runtime", "max_iterations", 2.5),
        ("runtime", "max_iterations", True),
        ("runtime", "workspace", ""),
        ("runtime", "workspace", "   "),
        ("runtime", "workspace", None),
        ("runtime", "workspace", 123),
    ],
)
def test_rejects_invalid_field(
    tmp_path: Path, section: str, field: str, value: Any
) -> None:
    with pytest.raises(ConfigError, match=rf"{section}\.{field}"):
        load_config(_write(tmp_path, _with(section, field, value)))


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("factory", "name"),
        ("factory", "environment"),
        ("llm", "provider"),
        ("llm", "model"),
        ("runtime", "workspace"),
        ("runtime", "max_iterations"),
    ],
)
def test_rejects_missing_field(tmp_path: Path, section: str, field: str) -> None:
    data = {name: dict(values) for name, values in VALID_CONFIG.items()}
    del data[section][field]

    with pytest.raises(ConfigError, match=rf"{section}\.{field}: Field required"):
        load_config(_write(tmp_path, data))


def test_rejects_unknown_field(tmp_path: Path) -> None:
    data = _with("runtime", "max_iteration", 5)  # typo

    with pytest.raises(ConfigError, match=r"runtime\.max_iteration: Extra inputs"):
        load_config(_write(tmp_path, data))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")


def test_path_is_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="is a directory"):
        load_config(tmp_path)


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "factory.yaml"
    path.write_text("factory: [unclosed\n")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "factory.yaml"
    path.write_text("")

    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_top_level_not_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must contain a mapping"):
        load_config(_write(tmp_path, ["not", "a", "mapping"]))
