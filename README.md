# skylab

Autonomous pretraining research tool. Define an experiment, pick a search strategy, set a time budget, and walk away. Skylab runs the loop — propose modifications, train for 5 minutes, measure, keep or discard — and you come back to a log of experiments and a better model.

Fork of [@karpathy's autoresearch](https://github.com/karpathy/autoresearch), rebuilt as a self-contained orchestration tool with pluggable search strategies and execution backends.

## How it works

Skylab separates the **tool** (orchestration, strategies, tracking) from the **experiment** (model, data, training):

```
skylab/                       # The orchestration tool
├── orchestrator.py           # Core loop: propose → train → evaluate → keep/revert
├── strategy/                 # Pluggable search strategies
│   ├── llm.py                # Claude Code modifies train.py autonomously
│   └── sweep.py              # Grid/random search over hyperparameter constants
├── runner/                   # Execution backends
│   ├── local.py              # Local GPU
│   └── remote.py             # Remote GPU via SSH
├── db.py                     # SQLite experiment database
├── monitor/server.py         # Live dashboard
└── cli.py                    # CLI entry point

experiments/gpt-pretrain/     # The experiment substrate
├── train.py                  # Model + optimizer + loop (strategies modify this)
├── prepare.py                # Data, tokenizer, evaluation (frozen — never modified)
├── schedules.py              # LR/momentum/decay schedules
└── experiment.toml           # Search surface, metric, constraints
```

The metric is **val_bpb** (validation bits per byte) — lower is better, vocab-size-independent. Each trial trains for a fixed 5-minute time budget.

## Quick start

**Requirements:** NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Download data and train tokenizer (one-time, ~2 min)
uv run skylab prepare

# Run a single experiment to verify setup
uv run experiments/gpt-pretrain/train.py
```

## Running experiments

```bash
# LLM-guided search (Claude Code proposes modifications autonomously)
skylab run --strategy llm --budget 8h --tag mar28

# Hyperparameter sweep
skylab run --strategy sweep --max-trials 50

# Remote GPU execution
skylab run --runner remote --budget 12h

# View experiment history
skylab history
skylab history --best
skylab history --export results.tsv
```

### Search strategies

**LLM strategy** (`--strategy llm`): Spawns Claude Code with experiment history and current code. Claude reads train.py, proposes a modification, and edits the file directly. The orchestrator diffs before/after, runs training, and keeps or reverts based on val_bpb. This is the autonomous overnight workflow.

**Sweep strategy** (`--strategy sweep`): Grid or random search over module-level constants (e.g., `DEPTH`, `MATRIX_LR`, `ASPECT_RATIO`). Configure parameters in `experiment.toml`. Good for systematic exploration of a known hyperparameter space.

### Execution backends

**Local** (`--runner local`): Runs training on the local GPU via subprocess.

**Remote** (`--runner remote`): Syncs code to a remote GPU host via SSH + rsync, runs training, streams results back. Configure `remote.toml` with host, user, key_path, and num_gpus. Multi-GPU uses `torchrun` automatically.

## Experiment configuration

Each experiment directory contains an `experiment.toml`:

```toml
[experiment]
name = "gpt-pretrain"

[search]
editable_files = ["train.py"]      # What strategies can modify
frozen_files = ["prepare.py"]      # Read-only context
metric = "val_bpb"
direction = "minimize"

[execution]
command = "uv run train.py"
time_budget_seconds = 300

[constraints]
max_vram_gb = 48.0
max_trial_wall_time = 600
```

## Experiment tracking

Results are stored in a SQLite database (`skylab.db`) in the experiment directory. Each trial records:

- Git commit hash and parent commit
- val_bpb and peak VRAM
- Status (success, crash, timeout)
- Code diff from parent
- Strategy that produced it
- Whether it was kept or reverted

Export to the legacy TSV format with `skylab history --export results.tsv`.

## Development

```bash
make sync          # Install dev dependencies
make test          # Run pytest
make lint          # Ruff check
make format        # Ruff format
make typecheck     # mypy
make standardize   # Format + lint autofix
```

## Design principles

- **Strategy/execution separation.** Strategies propose, runners execute, the orchestrator decides. These never bleed into each other (inspired by Ray Tune/Optuna).
- **The LLM doesn't control the search policy.** The orchestrator decides keep/revert based on the metric. The LLM's job is code generation, not search orchestration (inspired by AIDE).
- **Git as the versioning layer.** Each trial is a commit. Revert = `git reset`. History = `git log`.
- **Fixed time budget.** 5 minutes per trial. Makes results directly comparable regardless of what the strategy changes.

## License

MIT
