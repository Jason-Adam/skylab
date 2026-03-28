# Skylab

Autonomous pretraining research tool with pluggable search strategies.

## Architecture

Two layers — the **tool** (`skylab/`) and the **experiment substrate** (`experiments/`):

- `skylab/` — orchestration package: CLI, orchestrator loop, strategies, runners, DB
- `experiments/gpt-pretrain/` — the experiment being optimized (train.py is the editable surface)
- `experiments/gpt-pretrain/prepare.py` is FROZEN — never modify it (owns the eval metric)

## Commands

```bash
make sync          # Install deps (uv sync --group dev)
make test          # pytest
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy tests/ experiments/gpt-pretrain/schedules.py skylab/
make standardize   # format + lint autofix

skylab run --strategy llm --budget 8h    # LLM-guided autonomous research
skylab run --strategy sweep              # Hyperparameter sweep
skylab history                           # View experiment results
skylab history --best                    # Best trial
```

## Testing

```bash
uv run pytest                 # All tests
uv run pytest tests/test_db.py  # DB layer only
```

Tests don't require a GPU. `test_schedules.py` adds `experiments/gpt-pretrain/` to sys.path to import schedules.py.

## Key Constraints

- `experiments/gpt-pretrain/prepare.py` is read-only — it contains the fixed evaluation harness (`evaluate_bpb`)
- `train.py` runs at import time (no `main()` guard) — it cannot be imported without running training
- Strategies must implement the `Strategy` protocol in `skylab/strategy/base.py`
- Runners must implement the `Runner` protocol in `skylab/runner/base.py`
- The LLM strategy invokes Claude Code via `claude -p` subprocess
- `experiment.toml` declares the search surface, metric, and constraints for each experiment

## Code Style

- Python 3.10+, `from __future__ import annotations` in all skylab/ files
- ruff (I, F, S, B rules), mypy strict on tests/ and skylab/
- Dataclasses for config/state objects, no Pydantic
- Pre-commit hooks: ruff-format, ruff-lint, mypy
