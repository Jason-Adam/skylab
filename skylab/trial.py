"""Trial types for experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

Status = Literal["success", "crash", "timeout", "killed"]


@dataclass
class Proposal:
    """A proposed modification from a strategy."""

    description: str
    rationale: str
    modified_files: dict[str, str] = field(default_factory=dict)
    """Mapping of relative file path -> new file content."""


@dataclass
class TrialResult:
    """The outcome of a single experiment trial."""

    id: int | None = None
    experiment: str = ""
    commit: str = ""
    parent_commit: str | None = None
    val_bpb: float | None = None
    peak_vram_mb: float | None = None
    status: Status = "success"
    description: str = ""
    code_diff: str = ""
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy: str = ""
    kept: bool = False


@dataclass
class ExperimentConfig:
    """Configuration loaded from experiment.toml."""

    name: str = ""
    description: str = ""

    # Search surface
    editable_files: list[str] = field(default_factory=lambda: ["train.py"])
    frozen_files: list[str] = field(default_factory=lambda: ["prepare.py"])

    # Metric
    metric: str = "val_bpb"
    direction: Literal["minimize", "maximize"] = "minimize"
    metric_pattern: str = r"^val_bpb:\s+([\d.]+)"

    # Execution
    command: str = "uv run train.py"
    log_file: str = "run.log"
    time_budget_seconds: int = 300

    # Constraints
    max_vram_gb: float = 48.0
    max_trial_wall_time: int = 600

    # Strategy-specific config
    strategy_config: dict[str, object] = field(default_factory=dict)
