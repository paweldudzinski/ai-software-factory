from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from software_factory.infrastructure.config.schema import FactoryConfig


class ConfigError(Exception):
    """Raised when the factory config cannot be loaded or is invalid."""


def load_config(path: Path) -> FactoryConfig:
    """Read, parse and validate the factory config file."""
    text = _read_pain_text(path)
    data = _load_yaml(text)

    try:
        return FactoryConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid config in {path} ({exc.error_count()} error(s)):"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"]
        if error["type"] not in ("missing", "extra_forbidden"):
            message += f" (got {error['input']!r})"
        lines.append(f"  - {location}: {message}")
    return "\n".join(lines)


def _read_pain_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    except IsADirectoryError:
        raise ConfigError(f"Config path is a directory, not a file: {path}") from None
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc.strerror}") from exc


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML\n{exc}") from exc

    if data is None:
        raise ConfigError("Config file is empty")
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file must contain a mapping at the top level, "
            f"got {type(data).__name__}"
        )
    return data
