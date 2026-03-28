"""LLM strategy — code modification via Claude Code CLI.

Spawns `claude -p` with a carefully crafted prompt containing:
- The current train.py code
- Summarized experiment history
- The experiment constraints and goals

Claude Code reads/writes train.py directly in the experiment directory.
The orchestrator diffs before/after to capture what changed.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from skylab.context import summarize_history
from skylab.trial import ExperimentConfig, Proposal, TrialResult

logger = logging.getLogger(__name__)

# Maximum number of trials before the LLM strategy gives up
DEFAULT_MAX_TRIALS = 200

SYSTEM_PROMPT = """\
You are an autonomous ML researcher optimizing a GPT pretraining script.
Your goal: minimize val_bpb (validation bits per byte).

## Rules
- You may ONLY modify the file(s) listed under "Editable files" below.
- Do NOT modify prepare.py or any frozen files — they contain the fixed evaluation harness.
- The training runs for a fixed 5-minute time budget. You cannot change this.
- VRAM is a soft constraint — some increase is acceptable for meaningful val_bpb gains.
- Simplicity criterion: all else being equal, simpler is better. A small improvement
  that adds ugly complexity is not worth it. Removing code and getting equal or better
  results is a great outcome.

## What you can change
- Model architecture (layers, heads, embeddings, attention patterns)
- Optimizer settings (learning rates, weight decay, momentum schedules)
- Hyperparameters (batch size, model depth/width, aspect ratio)
- Training loop details (gradient accumulation, compilation settings)
- Anything in train.py that might improve val_bpb

## Your task
Based on the experiment history below, make ONE focused modification to train.py
that you believe will improve val_bpb. Make the change directly — edit the file.

After editing, print a single line describing what you changed, like:
CHANGE: <description of what you changed and why>
"""


def _build_prompt(
    history: list[TrialResult],
    editable_files: dict[str, str],
    config: ExperimentConfig,
    max_context_trials: int = 20,
) -> str:
    """Build the prompt for Claude Code."""
    parts: list[str] = []

    # History summary
    summary = summarize_history(history, max_trials=max_context_trials)
    parts.append(summary)

    # Frozen file names (so the LLM knows what not to touch)
    parts.append(f"\n## Frozen files (DO NOT MODIFY): {', '.join(config.frozen_files)}")

    # Editable files with current content
    parts.append(f"\n## Editable files: {', '.join(config.editable_files)}")
    parts.append(
        "The editable files are already in your working directory. Read and modify them directly."
    )

    # Goal
    parts.append(f"\n## Goal: {config.direction} {config.metric}")
    parts.append(f"Current best: {_best_metric(history, config)}")

    parts.append("\nMake ONE focused change. Edit train.py now.")

    return "\n".join(parts)


def _best_metric(history: list[TrialResult], config: ExperimentConfig) -> str:
    kept = [t for t in history if t.kept and t.val_bpb is not None]
    if not kept:
        return "no successful trials yet"
    if config.direction == "minimize":
        best = min(kept, key=lambda t: t.val_bpb or float("inf"))
    else:
        best = max(kept, key=lambda t: t.val_bpb or float("-inf"))
    return f"val_bpb = {best.val_bpb:.6f}"


class LLMStrategy:
    """Search strategy that uses Claude Code to propose code modifications.

    Spawns `claude -p` as a subprocess in the experiment directory.
    Claude Code reads train.py, modifies it, and the orchestrator
    captures the diff.
    """

    def __init__(self, llm_config: dict | None = None) -> None:
        self._config = llm_config or {}
        self._max_trials = self._config.get("max_trials", DEFAULT_MAX_TRIALS)
        self._max_context_trials = self._config.get("max_context_trials", 20)
        self._model = self._config.get("model")

    def propose(
        self,
        history: list[TrialResult],
        editable_files: dict[str, str],
        config: ExperimentConfig,
    ) -> Proposal:
        prompt = _build_prompt(
            history, editable_files, config, self._max_context_trials
        )

        # Snapshot editable files before Claude Code modifies them
        experiment_dir = self._find_experiment_dir(config)
        snapshots = {}
        for filename in config.editable_files:
            path = experiment_dir / filename
            if path.exists():
                snapshots[filename] = path.read_text()

        # Run Claude Code
        output = self._invoke_claude(prompt, experiment_dir)

        # Read modified files
        modified_files: dict[str, str] = {}
        for filename in config.editable_files:
            path = experiment_dir / filename
            if path.exists():
                new_content = path.read_text()
                if new_content != snapshots.get(filename, ""):
                    modified_files[filename] = new_content

        if not modified_files:
            # Claude didn't modify anything — restore and raise
            for filename, content in snapshots.items():
                (experiment_dir / filename).write_text(content)
            raise RuntimeError("LLM strategy did not modify any files")

        # Extract the CHANGE: description from output
        description = _extract_change_description(output)

        return Proposal(
            description=description,
            rationale=f"LLM-proposed modification (trial {len(history) + 1})",
            modified_files=modified_files,
        )

    def should_continue(self, history: list[TrialResult]) -> bool:
        return len(history) < self._max_trials

    def _find_experiment_dir(self, config: ExperimentConfig) -> Path:
        """Find the experiment directory from the config name."""
        # Try common locations
        candidates = [
            Path("experiments") / config.name,
            Path("experiments/gpt-pretrain"),
            Path("."),
        ]
        for candidate in candidates:
            if (candidate / "experiment.toml").exists():
                return candidate
        return Path(".")

    def _invoke_claude(self, prompt: str, cwd: Path) -> str:
        """Invoke Claude Code CLI and return its output."""
        cmd = ["claude", "-p", prompt, "--allowedTools", "Edit,Read,Write,Bash"]

        if self._model:
            cmd.extend(["--model", self._model])

        logger.info("Invoking Claude Code...")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout for LLM response
                env={"CLAUDE_CODE_ENTRYPOINT": "skylab-strategy"},
            )
            if result.returncode != 0:
                logger.warning("Claude Code exited with code %d", result.returncode)
                logger.debug("stderr: %s", result.stderr)
            return result.stdout
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Claude Code timed out (5 min)") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "Claude Code CLI not found. Install it: npm install -g @anthropic-ai/claude-code"
            ) from e


def _extract_change_description(output: str) -> str:
    """Extract the CHANGE: line from Claude Code output."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CHANGE:"):
            return stripped[7:].strip()
    # Fallback: use the last non-empty line
    for line in reversed(output.splitlines()):
        if line.strip():
            return line.strip()[:100]
    return "LLM modification (no description captured)"
