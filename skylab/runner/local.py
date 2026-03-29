"""Local execution backend — supports CUDA, MPS, and CPU."""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from skylab.runner.base import RunResult

logger = logging.getLogger(__name__)


def _detect_local_device() -> str:
    """Detect the best available local device."""
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for MPS (Apple Silicon)
    try:
        result = subprocess.run(
            ["python3", "-c", "import torch; print(torch.backends.mps.is_available())"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip() == "True":
            return "mps"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "cpu"


class LocalRunner:
    """Runs training locally via subprocess."""

    def execute(
        self,
        experiment_dir: str,
        command: str,
        log_file: str,
        timeout: int,
    ) -> RunResult:
        exp_path = Path(experiment_dir)
        log_path = exp_path / log_file

        start = time.monotonic()
        timed_out = False

        # Pass SKYLAB_DEVICE through to the subprocess if set
        env = os.environ.copy()

        try:
            result = subprocess.run(
                shlex.split(command),
                cwd=str(exp_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as e:
            exit_code = -1
            stdout = (e.stdout or b"").decode(errors="replace")
            stderr = (e.stderr or b"").decode(errors="replace")
            timed_out = True

        duration = time.monotonic() - start

        # Write combined output to log file
        log_path.write_text(stdout + stderr)

        # Extract metrics from output
        metrics = extract_metrics(stdout + stderr)

        return RunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=timed_out,
            metrics=metrics,
        )

    def check(self) -> bool:
        """Check if a compute device is available locally (CUDA, MPS, or CPU)."""
        device = _detect_local_device()
        logger.info("Local device: %s", device)
        return True  # CPU is always available


def extract_metrics(output: str) -> dict[str, float]:
    """Extract key=value metrics from the last training output summary block.

    Looks for the --- sentinel and parses lines like:
        val_bpb:          0.997900
        peak_vram_mb:     45060.2

    If multiple --- blocks exist, only the last one is used.
    """
    metrics: dict[str, float] = {}
    in_summary = False

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "---":
            in_summary = True
            metrics.clear()  # Reset — only keep the last block
            continue
        if not in_summary:
            continue
        match = re.match(r"^(\w+):\s+([\d.]+)", stripped)
        if match:
            key = match.group(1)
            try:
                metrics[key] = float(match.group(2))
            except ValueError:
                pass

    return metrics
