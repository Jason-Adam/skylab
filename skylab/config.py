"""Load experiment configuration from experiment.toml."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from skylab.trial import ExperimentConfig


def load_experiment_config(experiment_dir: Path) -> ExperimentConfig:
    """Parse experiment.toml into an ExperimentConfig."""
    toml_path = experiment_dir / "experiment.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"No experiment.toml found in {experiment_dir}")

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    config = ExperimentConfig()
    config.experiment_dir = str(experiment_dir.resolve())

    # [experiment] section
    exp = data.get("experiment", {})
    config.name = exp.get("name", experiment_dir.name)
    config.description = exp.get("description", "")

    # [search] section
    search = data.get("search", {})
    config.editable_files = search.get("editable_files", config.editable_files)
    config.frozen_files = search.get("frozen_files", config.frozen_files)
    config.metric = search.get("metric", config.metric)
    config.direction = search.get("direction", config.direction)
    config.metric_pattern = search.get("metric_pattern", config.metric_pattern)

    # [execution] section
    exe = data.get("execution", {})
    config.command = exe.get("command", config.command)
    config.log_file = exe.get("log_file", config.log_file)
    config.time_budget_seconds = int(
        exe.get("time_budget_seconds", config.time_budget_seconds)
    )

    # [constraints] section
    constraints = data.get("constraints", {})
    config.max_vram_gb = float(constraints.get("max_vram_gb", config.max_vram_gb))
    config.max_trial_wall_time = int(
        constraints.get("max_trial_wall_time", config.max_trial_wall_time)
    )

    # Strategy-specific sections (strategy.llm, strategy.sweep, etc.)
    strategy = data.get("strategy", {})
    for key, value in strategy.items():
        if isinstance(value, dict):
            config.strategy_config[key] = value

    return config
