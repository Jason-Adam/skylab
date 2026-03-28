.PHONY: sync test lint format typecheck standardize train prepare monitor remote check-gpu

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
	uv run mypy tests/ schedules.py

standardize:
	uv run ruff format . && uv run ruff check --fix .

train:
	uv run train.py

prepare:
	uv run prepare.py

monitor:
	uv run monitor.py

remote:
	bash remote_run.sh

check-gpu:
	bash remote_run.sh --check
