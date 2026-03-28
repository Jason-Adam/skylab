.PHONY: sync test lint format typecheck standardize run run-sweep run-remote history monitor prepare check-gpu

sync:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy tests/ experiments/gpt-pretrain/schedules.py skylab/

standardize:
	uv run ruff format . && uv run ruff check --fix .

# --- Skylab orchestrator commands ---

run:
	uv run skylab run

run-sweep:
	uv run skylab run --strategy sweep

run-remote:
	uv run skylab run --runner remote

history:
	uv run skylab history

monitor:
	uv run skylab monitor

prepare:
	uv run skylab prepare

check-gpu:
	uv run skylab check-gpu --runner remote
