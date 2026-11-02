# ABOUTME: Build, test, and run entry points for deslop.
# ABOUTME: Use these targets instead of invoking tools directly.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: install test cov lint fmt typecheck check build docs docs-serve example llm-reference english-generic registers register-fixtures

install:
	uv sync --all-groups

test:
	uv run pytest

cov:
	uv run pytest --cov=deslop --cov-report=term-missing --cov-fail-under=85

lint:
	uv run ruff check src tests scripts examples
	uv run ruff format --check src tests scripts examples

fmt:
	uv run ruff format src tests scripts examples

typecheck:
	uv run mypy

check: lint typecheck cov

build:
	rm -rf dist && uv build

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

example:
	uv run python examples/apply_example.py

llm-reference:
	uv run python scripts/build_llm_reference.py

english-generic:
	uv run python scripts/build_english_generic.py .cache/gutenberg

register-fixtures:
	uv run python scripts/build_registers.py $(GUTENBERG_DIR)
	uv run python scripts/build_register_slop.py

# Fans out over four documents: needs CONFIRM=1 and MAX_COST (USD); the projection prints first.
registers:
	uv run python scripts/run_registers.py $(or $(BUDGET),5) $(or $(N),3) --max-cost $(or $(MAX_COST),0) $(if $(filter 1,$(CONFIRM)),--confirm,) $(ONLY)
