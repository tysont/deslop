# ABOUTME: Build, test, and run entry points for deslop.
# ABOUTME: Use these targets instead of invoking tools directly.
.PHONY: install test cov lint fmt fixtures llm-reference english-generic

install:
	uv sync

test:
	uv run pytest

cov:
	uv run pytest --cov=deslop --cov-report=term-missing

lint:
	uv run ruff check src tests scripts

fmt:
	uv run ruff format src tests scripts

llm-reference:
	uv run python scripts/build_llm_reference.py

english-generic:
	uv run python scripts/build_english_generic.py .cache/gutenberg
