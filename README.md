# AI Software Factory

An AI-driven software factory: a runtime that processes tasks with LLM agents (Anthropic).

> **Status:** early development. The configuration layer, the `Task` domain model and the
> `factory` CLI are in place; agents and the task queue are not implemented yet.

## Requirements

- Python 3.13+

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the package in editable mode, the development dependencies (pytest) and the
`factory` command.

## Usage

Run commands from the project root (the default config path is relative to the current
directory):

```bash
factory info
```

```
AI Software Factory
-------------------
Version:       0.1.0
Environment:   local
Provider:      anthropic
Runtime:       ready
Active agents: 0
Active tasks:  0
```

Use a different config file:

```bash
factory info --config path/to/factory.yaml
```

List all commands with `factory --help`.

## Configuration

The factory is configured with `config/factory.yaml`:

```yaml
factory:
  name: software-factory
  environment: local

llm:
  provider: anthropic
  model: YOUR_MODEL

runtime:
  workspace: ./workspace
  max_iterations: 10
```

The config is validated on load. If any rule is broken, the factory does not start and every
error is reported:

| Field                    | Rule                                      |
|--------------------------|-------------------------------------------|
| `factory.name`           | non-empty string                          |
| `factory.environment`    | `local`                                   |
| `llm.provider`           | `anthropic`                               |
| `llm.model`              | non-empty string                          |
| `runtime.workspace`      | non-empty path                            |
| `runtime.max_iterations` | integer greater than 0 (`"10"` is rejected) |

All fields are required and unknown keys are rejected, so a typo such as `max_iteration`
is reported instead of silently ignored.

Example of an invalid config:

```
Invalid config in config/factory.yaml (2 error(s)):
  - llm.model: String should have at least 1 character (got '')
  - runtime.max_iterations: Input should be greater than 0 (got -10)
```

Keep secrets such as API keys in a `.env` file (ignored by git), not in `factory.yaml`.

## Project structure

```
config/
  factory.yaml                # factory configuration
src/software_factory/
  application/                # use cases (not implemented yet)
  cli/
    app.py                    # `factory` CLI (Typer)
  domain/
    task.py                   # Task model and TaskStatus
  infrastructure/
    config/
      schema.py               # Pydantic models describing a valid config
      loader.py               # load_config(): read YAML, validate, return FactoryConfig
tests/                        # pytest test suite
```

## Development

Run the tests:

```bash
pytest
```

Use `pytest -v` for one line per test.
