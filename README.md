# autoresearch

![teaser](progress.png)

Autonomous LLM training research — give an AI agent a real training setup and let it experiment overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and a better model.

Based on [@karpathy's autoresearch](https://github.com/karpathy/autoresearch), extended with remote GPU execution and multi-GPU distributed training support.

## How it works

Three files matter:

- **`prepare.py`** — fixed constants, data prep, tokenizer, dataloader, evaluation. Not modified.
- **`train.py`** — model, optimizer, training loop. **The agent edits this file**.
- **`program.md`** — agent instructions. **The human edits this file**.

Training runs for a **fixed 5-minute time budget**. The metric is **val_bpb** (validation bits per byte) — lower is better, vocab-size-independent so architectural changes are fairly compared. ~12 experiments/hour, ~100 overnight.

## Quick start

**Requirements:** NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# Run a single training experiment (~5 min)
uv run train.py
```

## Running the agent

Point your agent (Claude, Codex, etc.) at this repo and prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The agent reads `program.md`, sets up a branch, and loops: modify `train.py` → train → measure → keep or discard.

## Remote execution

Run experiments on a remote GPU machine without changing any training code. `remote_run.sh` syncs your code via SSH, runs training, and streams results back:

```bash
# Configure your remote host
vi remote.toml

# Verify connectivity and GPU
bash remote_run.sh --check

# Run an experiment remotely (same interface as local)
bash remote_run.sh > run.log 2>&1
grep "^val_bpb:" run.log
```

The script handles SSH connectivity, code sync (rsync), data preparation on first run, and timeout enforcement on the remote side. A `Dockerfile` is included for reproducible environments (CUDA 12.8, Python 3.10, pinned dependencies).

Configuration lives in `remote.toml` (gitignored):
- `host` — SSH hostname or IP
- `user` — SSH user (default: `ubuntu`)
- `key_path` — path to SSH key
- `workspace` — remote working directory
- `num_gpus` — set >1 for multi-GPU (uses `torchrun` automatically)

## Multi-GPU distributed training

`train.py` supports multi-GPU training via `torchrun`. Each GPU reads a disjoint subset of training data shards, gradients are averaged across GPUs before the optimizer step. The effective batch size scales linearly with GPU count.

```bash
# Local multi-GPU (e.g. 4 GPUs)
uv run torchrun --nproc_per_node=4 train.py

# Remote multi-GPU (set num_gpus in remote.toml)
bash remote_run.sh > run.log 2>&1
```

- **Backward compatible** — `uv run train.py` still works for single GPU, unchanged.
- **Same output format** — only rank 0 prints and runs evaluation. `grep "^val_bpb:" run.log` works identically.
- **Data requirement** — need at least as many training shards as GPUs. Download more with `uv run prepare.py --num-shards N`.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
remote_run.sh   — remote GPU execution script
remote.toml     — remote host configuration (gitignored)
Dockerfile      — reproducible CUDA environment
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Keeps scope manageable and diffs reviewable.
- **Fixed time budget.** 5 minutes per experiment. Makes results directly comparable regardless of what the agent changes.
- **Self-contained.** PyTorch and a few small packages. Multi-GPU via `torchrun`, single-GPU by default.

## License

MIT
