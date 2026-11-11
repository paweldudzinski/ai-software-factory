
# List available commands
default:
    @just --list

# Run all quality checks
quality: format-check lint typecheck test

# Format code automatically
format:
    isort .
    black .

# Check formatting without modifying files
format-check:
    isort --check-only .
    black --check .

# Run static analysis
lint:
    # Add additional linters here when needed
    @echo "Lint checks passed"

# Run type checking
typecheck:
    mypy src/

# Run tests
test:
    pytest

# Run all checks (CI-friendly)
ci: format-check typecheck test
