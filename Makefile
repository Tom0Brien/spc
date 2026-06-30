.PHONY: build rebuild format format-python format-cpp install-dev

# First-time install: resolves dependencies and compiles the C++ backend.
build:
	uv pip install -e .

# Force a recompile after editing C++ sources (a plain reinstall is a no-op).
# Assumes dependencies are already installed (run `make build` first).
rebuild:
	uv pip install -e . --force-reinstall --no-deps

install-dev:
	uv pip install -e ".[dev]"

format-python:
	uv run ruff format .
	uv run ruff check --fix .

format-cpp:
	find src include tests -type f \( -name "*.cc" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) | xargs uv run clang-format -i

format: format-python format-cpp
	@echo "All code formatted successfully!"
