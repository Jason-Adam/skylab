"""Runner protocol — execution backends for training trials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RunResult:
    """Result of executing a training script."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


class Runner(Protocol):
    """Protocol for execution backends."""

    def execute(
        self,
        experiment_dir: str,
        command: str,
        log_file: str,
        timeout: int,
    ) -> RunResult:
        """Run the training script and return results."""
        ...

    def check(self) -> bool:
        """Verify the execution environment is ready."""
        ...
