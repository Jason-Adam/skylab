"""Orchestrator — the core experiment loop.

Encodes the research loop from program.md:
    propose → commit → execute → evaluate → keep/revert
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from skylab.config import load_experiment_config
from skylab.db import Database
from skylab.runner.base import Runner
from skylab.strategy.base import Strategy
from skylab.trial import ExperimentConfig, TrialResult

logger = logging.getLogger(__name__)


def run(
    experiment_dir: Path,
    strategy: Strategy,
    runner: Runner,
    db: Database,
    budget_hours: float | None = None,
    max_trials: int | None = None,
    tag: str | None = None,
) -> TrialResult | None:
    """Run the autonomous experiment loop.

    Args:
        experiment_dir: Path to the experiment directory (e.g. experiments/gpt-pretrain).
        strategy: The search strategy to use.
        runner: The execution backend.
        db: The experiment database.
        budget_hours: Maximum wall-clock hours to run. None = unlimited.
        max_trials: Maximum number of trials. None = unlimited (strategy decides).
        tag: Git branch tag (e.g. "mar28"). Creates branch skylab/<tag>.

    Returns:
        The best TrialResult, or None if no successful trials.
    """
    config = load_experiment_config(experiment_dir)
    exp_dir_str = str(experiment_dir)

    # Set up git branch if tag provided
    if tag:
        _setup_branch(experiment_dir, tag)

    # Verify execution environment
    if not runner.check():
        logger.warning("Runner check failed — execution environment may not be ready")

    # 1. Establish baseline (first run, no modifications)
    logger.info("Running baseline trial...")
    baseline = _run_trial(runner, exp_dir_str, config, "baseline (no modifications)")
    baseline.experiment = config.name
    baseline.strategy = "baseline"
    baseline.kept = True
    db.record(baseline)

    if baseline.status != "success":
        logger.error("Baseline failed (%s) — cannot continue", baseline.status)
        return None

    best = baseline
    logger.info("Baseline val_bpb: %.6f", best.val_bpb)

    deadline = time.time() + budget_hours * 3600 if budget_hours else float("inf")
    trial_count = 0

    while True:
        # Check termination conditions
        if time.time() >= deadline:
            logger.info("Budget exhausted (%.1f hours)", budget_hours)
            break
        if max_trials and trial_count >= max_trials:
            logger.info("Max trials reached (%d)", max_trials)
            break

        history = db.history(config.name)
        if not strategy.should_continue(history):
            logger.info("Strategy has no more ideas")
            break

        # Read current editable files
        editable_files = _read_editable_files(experiment_dir, config)

        # Get proposal from strategy
        try:
            proposal = strategy.propose(history, editable_files, config)
        except Exception:
            logger.exception("Strategy failed to produce a proposal")
            break

        logger.info("Trial %d: %s", trial_count + 1, proposal.description)

        # Get current commit (before modification)
        parent_commit = _git_head(experiment_dir)

        # Apply proposal (write modified files)
        for filename, content in proposal.modified_files.items():
            (experiment_dir / filename).write_text(content)

        # Git commit the changes
        commit = _git_commit(experiment_dir, proposal.description)

        # Execute trial
        result = _run_trial(runner, exp_dir_str, config, proposal.description)
        result.experiment = config.name
        result.commit = commit
        result.parent_commit = parent_commit
        result.strategy = strategy.__class__.__name__
        result.code_diff = _git_diff(experiment_dir, parent_commit)

        # Decide: keep or revert
        if _is_improvement(result, best, config):
            result.kept = True
            db.record(result)
            logger.info(
                "KEEP: val_bpb %.6f -> %.6f (improvement of %.6f)",
                best.val_bpb,
                result.val_bpb,
                best.val_bpb - result.val_bpb,
            )
            best = result
        else:
            result.kept = False
            db.record(result)
            reason = result.status if result.status != "success" else "no improvement"
            logger.info(
                "REVERT (%s): val_bpb %s vs best %.6f",
                reason,
                f"{result.val_bpb:.6f}" if result.val_bpb else "N/A",
                best.val_bpb,
            )
            _git_revert(experiment_dir, parent_commit)

        trial_count += 1

    logger.info(
        "Done. %d trials, best val_bpb: %.6f (commit %s)",
        trial_count + 1,  # +1 for baseline
        best.val_bpb,
        best.commit,
    )
    return best


def _run_trial(
    runner: Runner,
    experiment_dir: str,
    config: ExperimentConfig,
    description: str,
) -> TrialResult:
    """Execute a single trial and extract results."""
    run_result = runner.execute(
        experiment_dir=experiment_dir,
        command=config.command,
        log_file=config.log_file,
        timeout=config.max_trial_wall_time,
    )

    # Determine status
    if run_result.timed_out:
        status = "timeout"
    elif run_result.exit_code != 0:
        status = "crash"
    else:
        status = "success"

    val_bpb = run_result.metrics.get("val_bpb")
    peak_vram = run_result.metrics.get("peak_vram_mb")

    return TrialResult(
        commit=_git_head_from_dir(experiment_dir),
        val_bpb=val_bpb,
        peak_vram_mb=peak_vram,
        status=status,
        description=description,
        duration_seconds=run_result.duration_seconds,
    )


def _is_improvement(
    result: TrialResult,
    best: TrialResult,
    config: ExperimentConfig,
) -> bool:
    """Determine if a trial result is an improvement over the current best."""
    if result.status != "success" or result.val_bpb is None:
        return False
    if best.val_bpb is None:
        return True
    if config.direction == "minimize":
        return result.val_bpb < best.val_bpb
    return result.val_bpb > best.val_bpb


# --- Git helpers ---


def _setup_branch(experiment_dir: Path, tag: str) -> None:
    branch = f"skylab/{tag}"
    subprocess.run(
        ["git", "checkout", "-b", branch],
        cwd=str(experiment_dir),
        capture_output=True,
        check=True,
    )
    logger.info("Created branch %s", branch)


def _git_head(experiment_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(experiment_dir),
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_head_from_dir(experiment_dir: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=experiment_dir,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_commit(experiment_dir: Path, message: str) -> str:
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(experiment_dir),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(experiment_dir),
        capture_output=True,
        check=True,
    )
    return _git_head(experiment_dir)


def _git_revert(experiment_dir: Path, target_commit: str) -> None:
    subprocess.run(
        ["git", "reset", "--hard", target_commit],
        cwd=str(experiment_dir),
        capture_output=True,
        check=True,
    )


def _git_diff(experiment_dir: Path, parent_commit: str) -> str:
    result = subprocess.run(
        ["git", "diff", parent_commit, "HEAD"],
        cwd=str(experiment_dir),
        capture_output=True,
        text=True,
    )
    return result.stdout


def _read_editable_files(
    experiment_dir: Path, config: ExperimentConfig
) -> dict[str, str]:
    """Read all editable files into a dict."""
    files: dict[str, str] = {}
    for filename in config.editable_files:
        path = experiment_dir / filename
        if path.exists():
            files[filename] = path.read_text()
    return files
