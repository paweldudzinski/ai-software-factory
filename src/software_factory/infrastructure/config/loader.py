from pathlib import Path

import yaml
from pydantic import ValidationError

from software_factory.infrastructure.config.schema import FactoryConfig


class ConfigError(Exception):
    """Raised when the factory config cannot be loaded or is invalid."""


def load_config(path: Path) -> FactoryConfig:
    """Read, parse and validate the factory config file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    except IsADirectoryError:
        raise ConfigError(f"Config path is a directory, not a file: {path}") from None
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc.strerror}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}:\n{exc}") from exc

    if data is None:
        raise ConfigError(f"Config file is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {path} must contain a mapping at the top level, "
            f"got {type(data).__name__}"
        )

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
